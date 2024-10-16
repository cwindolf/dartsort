from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
import numpy as np
import torch
from tqdm.auto import tqdm
from scipy.spatial import KDTree
from scipy.stats import pearsonr

from ..cluster import density
from dartsort.cluster.modes import smoothed_dipscore_at
from .colors import glasbey1024
from . import analysis_plots, layout, unit
from ..util.multiprocessing_util import CloudpicklePoolExecutor, get_pool, ThreadPoolExecutor
from ..util import spikeio
from .waveforms import geomplot

try:
    from ephysx import spike_interp
except ImportError:
    pass


# -- over_time_summary stuff


class GMMPlot(layout.BasePlot):
    width = 1
    height = 1
    kind = "gmm"

    def draw(self, panel, gmm, unit_id):
        raise NotImplementedError


class DPCSplitPlot(GMMPlot):
    kind = "triplet"
    width = 5
    height = 5

    def __init__(self, spike_kind="split", feature="pca", inherit_chans=True, common_chans=True, dist_vmax=1., cmap=plt.cm.rainbow):
        self.spike_kind = spike_kind
        assert feature in ("pca", "spread_amp")
        self.feature = feature
        self.inherit_chans = inherit_chans
        self.common_chans = common_chans
        self.dist_vmax = dist_vmax
        self.cmap = cmap
        self.show_values = True

    def draw(self, panel, gmm, unit_id):
        if self.feature == "pca":
            if self.spike_kind == "residual_full":
                (in_unit,) = (gmm.labels == unit_id).nonzero(as_tuple=True)
                features = torch.empty(
                    (in_unit.numel(), gmm.residual_pca_rank), device=gmm.device
                )
                for sl, data in gmm.batches(in_unit):
                    gmm[unit_id].residual_embed(**data, out=features[sl])
                z = features[:, : gmm.dpc_split_kw.rank].numpy(force=True)
            elif self.spike_kind == "split":
                _, in_unit, z = gmm.split_features(unit_id)
            elif self.spike_kind == "global":
                in_unit, data = gmm.get_training_data(unit_id)
                waveforms = gmm[unit_id].to_unit_channels(
                    waveforms=data["waveforms"],
                    times=data["times"],
                    waveform_channels=data["waveform_channels"],
                )
                loadings, mean, components, svs = spike_interp.fit_pcas(
                    waveforms.reshape(in_unit.numel(), -1),
                    missing=None,
                    empty=None,
                    rank=gmm.dpc_split_kw.rank,
                    show_progress=False,
                )
                z = loadings.numpy(force=True)
        elif self.feature == "spread_amp":
            assert self.spike_kind in ("split", "global")
            in_unit, data = gmm.get_training_data(unit_id)
            waveforms = data["waveforms"]
            channel_norms = torch.sqrt(torch.nan_to_num(waveforms.square().sum(1)))
            amp = channel_norms.max(1).values
            logs = torch.nan_to_num(channel_norms.log())
            spread = (channel_norms * logs).sum(1)
            z = np.c_[amp.numpy(force=True), spread.numpy(force=True)]
            z /= mad(z, 0)

        if in_unit is None:
            ax = panel.subplots()
            ax.set_title("no features")
            return

        in_unit = in_unit.numpy(force=True)
        zu, idx, inv = np.unique(z, return_index=True, return_inverse=True, axis=0)
        dens = density.density_peaks_clustering(
            zu,
            sigma_local=gmm.dpc_split_kw.sigma_local,
            n_neighbors_search=gmm.dpc_split_kw.n_neighbors_search,
            remove_clusters_smaller_than=int(gmm.min_cluster_size // 2),
            radius_search=gmm.dpc_split_kw.radius_search,
            return_extra=True,
        )
        if "density" not in dens:
            ax = panel.subplots()
            ax.set_title("all spikes binned")
            ax.scatter(*z[:, :2].T, s=5, lw=0, color="k")
            ax.autoscale_view()
            return

        ru = np.unique(dens["labels"])
        panel_top, panel_bottom = panel.subfigures(nrows=2, height_ratios=[.6, 1])
        panel_top, axes = analysis_plots.density_peaks_study(
            z,
            dens,
            s=10,
            idx=idx,
            inv=inv,
            fig=panel_top,
        )
        axes[-1].set_title(f"n={(ru>=0).sum()}", fontsize=8)
        axes[0].set_title(self.spike_kind)
        if self.feature == "spread_amp":
            axes[0].set_xlabel("spread")
            axes[0].set_ylabel("amp")

        axes = panel_bottom.subplots(ncols=3)
        axes = {'d': axes[0], 'f': axes[1], 'e': axes[2]}
        labels = dens['labels'][inv]
        in_unit = torch.from_numpy(in_unit)
        ids = np.unique(labels)
        ids = ids[ids >= 0]
        new_units = []
        chans_kw = {} 
        if self.inherit_chans:
            chans_kw = dict(
                channels=gmm[unit_id].channels,
                max_channel=gmm[unit_id].max_channel,
            )
        for j, label in enumerate(ids):
            u = spike_interp.InterpUnit(
                do_interp=False,
                **gmm.unit_kw,
            )
            inu = in_unit[np.flatnonzero(labels == label)]
            inu, train_data = gmm.get_training_data(
                unit_id,
                waveform_kind="original",
                in_unit=inu,
                sampling_method=gmm.sampling_method,
            )
            u.fit_center(
                **train_data,
                padded_geom=gmm.data.padded_registered_geom,
                show_progress=False,
                **chans_kw,
            )
            new_units.append(u)

        ju = [(j, u) for j, u in enumerate(new_units) if u.n_chans_unit]

        # plot new unit maxchan wfs and old one in black
        gmc = gmm[unit_id].max_channel
        for ax, pick in zip((axes["d"], axes["f"]), ("unit", "shared")):
            ax.axhline(0, c="k", lw=0.8)
            all_means = []
            for j, unit in ju:
                if unit.do_interp:
                    times = unit.interp.grid.squeeze()
                    if self.fitted_only:
                        times = times[unit.interp.grid_fitted]
                else:
                    times = torch.tensor([sum(gmm.t_bounds) / 2]).to(gmm.device)
                times = torch.atleast_1d(times)

                chans = torch.full((times.numel(),), unit.max_channel if pick == "unit" else gmc, device=times.device)
                means = unit.get_means(times).to(gmm.device)
                if j > 0:
                    all_means.append(means.mean(0))
                means = unit.to_waveform_channels(means, waveform_channels=chans[:, None])
                means = means[..., 0]
                means = gmm.data.tpca._inverse_transform_in_probe(means)
                means = means.numpy(force=True)
                color = glasbey1024[j]
    
                lines = np.stack(
                    (np.broadcast_to(np.arange(means.shape[1])[None], means.shape), means),
                    axis=-1,
                )
                ax.add_collection(LineCollection(lines, colors=color, lw=1))
            ax.autoscale_view()
            ax.set_xticks([])
            ax.spines[["top", "right", "bottom"]].set_visible(False)
            ax.set_title(pick)

        # plot distance matrix
        kind = gmm.merge_metric
        min_overlap = gmm.min_overlap
        subset_channel_index = None
        if gmm.merge_on_waveform_radius:
            subset_channel_index = gmm.data.registered_reassign_channel_index
        nu = len(new_units)
        divergences = torch.full((nu, nu), torch.nan)
        for i, ua in ju:
            # print(f"{i=} {ua.n_chans_unit=} {ua.channels.tolist()=}")
            for j, ub in ju:
                # print(f"{j=} {ub.n_chans_unit=} {ub.channels.tolist()=}")
                if i == j:
                    divergences[i, j] = 0
                    continue
                divergences[i, j] = ua.divergence(
                    ub,
                    kind=kind,
                    min_overlap=min_overlap,
                    subset_channel_index=subset_channel_index,
                    common_chans=self.common_chans,
                )
        dists = divergences.numpy(force=True)

        axis = axes["e"]
        im = axis.imshow(
            dists,
            vmin=0,
            vmax=self.dist_vmax,
            cmap=self.cmap,
            origin="lower",
            interpolation="none",
        )
        if self.show_values:
            for (j, i), label in np.ndenumerate(dists):
                axis.text(
                    i,
                    j,
                    f"{label:.2f}".lstrip("0"),
                    ha="center",
                    va="center",
                    clip_on=True,
                    fontsize=5,
                )
        panel.colorbar(im, ax=axis, shrink=0.3)
        axis.set_xticks(range(len(new_units)))
        axis.set_yticks(range(len(new_units)))
        for i, (tx, ty) in enumerate(
            zip(axis.xaxis.get_ticklabels(), axis.yaxis.get_ticklabels())
        ):
            tx.set_color(glasbey1024[i])
            ty.set_color(glasbey1024[i])
        axis.set_title(gmm.merge_metric)


class ZipperSplitPlot(GMMPlot):
    kind = "triplet"
    width = 5
    height = 5

    def __init__(
        self,
        cmap=plt.cm.rainbow,
        fitted_only=True,
        scaled=True,
        amplitude_scaling_std=np.sqrt(0.001),
        amplitude_scaling_limit=1.2,
        dist_vmax=1.0,
        show_values=True,
    ):
        self.cmap = cmap
        self.fitted_only = fitted_only
        self.scaled = scaled
        self.inv_lambda = 1.0 / (amplitude_scaling_std**2)
        self.scale_clip_low = 1.0 / amplitude_scaling_limit
        self.scale_clip_high = amplitude_scaling_limit
        self.dist_vmax = dist_vmax
        self.show_values = show_values
        self.title = "grid mean dists"
        if self.scaled:
            self.title = f"scaled {self.title}"

    def draw(self, panel, gmm, unit_id):
        in_unit = np.flatnonzero(gmm.labels == unit_id)
        times = gmm.data.times_seconds[in_unit].numpy(force=True)
        amps = np.nanmax(gmm.data.amp_vecs[in_unit], 1)
        z = np.c_[times / mad(times), amps / mad(amps)]
        dens = density.density_peaks_clustering(
            z,
            # sigma_local=gmm.dpc_split_kw.sigma_local,
            sigma_local=0.5,
            sigma_regional=1.0,
            min_bin_size=0.05,
            n_neighbors_search=gmm.dpc_split_kw.n_neighbors_search,
            remove_clusters_smaller_than=gmm.min_cluster_size,
            return_extra=True,
        )

        labels = dens["labels"]
        ids = np.unique(labels)
        ids = ids[ids >= 0]

        if ids.size > 1:
            top, bottom = panel.subfigures(nrows=2)
            axes_top = top.subplot_mosaic("abc")
            axes_bottom = bottom.subplot_mosaic("de")
            axes = {**axes_top, **axes_bottom}
        else:
            axes = panel.subplot_mosaic("abc")

        _, _ = analysis_plots.density_peaks_study(
            z,
            dens,
            s=10,
            fig=panel,
            axes=np.array([axes[k] for k in "abc"]),
        )
        if ids.size <= 1:
            return

        new_units = []
        for label in ids:
            u = spike_interp.InterpUnit(
                do_interp=False,
                **gmm.unit_kw,
            )
            inu = torch.tensor(in_unit[np.flatnonzero(labels == label)])
            inu, train_data = gmm.get_training_data(
                unit_id,
                waveform_kind="original",
                in_unit=inu,
                sampling_method=gmm.sampling_method,
            )
            u.fit_center(**train_data, show_progress=False)
            new_units.append(u)

        # plot new unit maxchan wfs and old one in black
        ax = axes["d"]
        ax.axhline(0, c="k", lw=0.8)
        all_means = []
        for j, unit in enumerate((gmm[unit_id], *new_units)):
            if unit.do_interp:
                times = unit.interp.grid.squeeze()
                if self.fitted_only:
                    times = times[unit.interp.grid_fitted]
            else:
                times = torch.tensor([sum(gmm.t_bounds) / 2]).to(gmm.device)
            times = torch.atleast_1d(times)

            chans = torch.full((times.numel(),), unit.max_channel, device=times.device)
            means = unit.get_means(times).to(gmm.device)
            if j > 0:
                all_means.append(means.mean(0))
            means = unit.to_waveform_channels(means, waveform_channels=chans[:, None])
            means = means[..., 0]
            means = gmm.data.tpca._inverse_transform_in_probe(means)
            means = means.numpy(force=True)
            color = "k"
            if j > 0:
                color = glasbey1024[j - 1]

            lines = np.stack(
                (np.broadcast_to(np.arange(means.shape[1])[None], means.shape), means),
                axis=-1,
            )
            ax.add_collection(LineCollection(lines, colors=color, lw=1))
        ax.autoscale_view()
        ax.set_xticks([])
        ax.spines[["top", "right", "bottom"]].set_visible(False)

        # plot distance matrix
        kind = gmm.merge_metric
        min_overlap = gmm.min_overlap
        subset_channel_index = None
        if gmm.merge_on_waveform_radius:
            subset_channel_index = gmm.data.registered_reassign_channel_index
        nu = len(new_units)
        divergences = torch.full((nu, nu), torch.nan)
        for i, ua in enumerate(range(nu)):
            for j, ub in enumerate(range(nu)):
                if ua == ub:
                    divergences[i, j] = 0
                    continue
                divergences[i, j] = new_units[ua].divergence(
                    new_units[ub],
                    kind=kind,
                    min_overlap=min_overlap,
                    subset_channel_index=subset_channel_index,
                )
        dists = divergences.numpy(force=True)

        axis = axes["e"]
        im = axis.imshow(
            dists,
            vmin=0,
            vmax=self.dist_vmax,
            cmap=self.cmap,
            origin="lower",
            interpolation="none",
        )
        if self.show_values:
            for (j, i), label in np.ndenumerate(dists):
                axis.text(i, j, f"{label:.2f}", ha="center", va="center", clip_on=True)
        panel.colorbar(im, ax=axis, shrink=0.3)
        axis.set_xticks(range(len(new_units)))
        axis.set_yticks(range(len(new_units)))
        for i, (tx, ty) in enumerate(
            zip(axis.xaxis.get_ticklabels(), axis.yaxis.get_ticklabels())
        ):
            tx.set_color(glasbey1024[i])
            ty.set_color(glasbey1024[i])
        axis.set_title(gmm.merge_metric)


class MStep(GMMPlot):
    kind = "single"
    width = 4
    height = 5
    alpha = 0.05

    def draw(self, panel, gmm, unit_id, unit=None, in_unit=None, axes=None):
        if unit is None:
            unit = gmm[unit_id]
        in_unit, utd = gmm.get_training_data(unit_id, in_unit=in_unit, waveform_kind="original")
        times = utd['times']
        waveforms = utd['waveforms']
        waveform_channels = utd['waveform_channels']
        padded_geom = gmm.data.padded_registered_geom

        if unit.impute_before_center:
            waveforms_rel = unit.impute(
                times,
                waveforms,
                waveform_channels,
                padded_registered_geom=padded_geom,
                centered=False,
            )
            waveforms_rel = waveforms_rel.reshape(
                len(waveforms_rel),
                -1,
                unit.channels_valid.numel(),
            )
        else:
            waveforms_rel = unit.to_unit_channels(
                waveforms,
                times,
                waveform_channels=waveform_channels,
                fill_mode="constant",
                constant_value=torch.nan,
            )
        n, r, c = waveforms_rel.shape
        x = waveforms_rel.permute(0, 2, 1).reshape(n, -1).numpy(force=True)

        mean = unit.mean.reshape(r, c).T.numpy(force=True).ravel()
        std = unit.std.reshape(r, c).T.numpy(force=True).ravel()
        domain = r * unit.channels_valid.numpy(force=True)[:, None] + np.arange(r)
        domain = domain.ravel()

        color = glasbey1024[unit_id % len(glasbey1024)]

        if axes is None:
            axes = panel.subplots(nrows=2, sharex=True)
        ax, ay = axes
        ax.plot(domain, x.T, color="k", alpha=self.alpha)
        ax.fill_between(
            domain,
            mean - std,
            mean + std,
            color=color,
            alpha=0.5,
            lw=0,
            zorder=11,
        )
        ax.plot(domain, mean, lw=1, color=color)
        ay.plot(domain, np.abs(mean), color=color, lw=1, label='fitted |mean|')
        ay.plot(domain, std, color=color, ls="--", lw=1, label='fitted std')
        ay.plot(domain, np.abs(np.nanmean(x, axis=0)), color='k', ls="--", lw=1, label='emp |mean|')
        ay.plot(domain, np.nanstd(x, axis=0), color='k', ls=":", lw=1, label='emp std')
        ay.legend(loc='upper left', frameon=False, fancybox=False)
        ay.set_xlabel("channel-major feature index")


class KMeansPPSPlitPlot(GMMPlot):
    kind = "triplet"
    width = 6
    height = 7

    def __init__(
        self,
        cmap=plt.cm.rainbow,
        fitted_only=True,
        amplitude_scaling_std=np.sqrt(0.001),
        amplitude_scaling_limit=1.2,
        merge_on_waveform_radius=True,
        dist_vmax=1.0,
        show_values=True,
        n_clust=None,
        n_iter=None,
        common_chans=False,
        inherit_chans=False,
        impute_before_center=False,
        with_proportions=False,
        min_overlap=0.0,
        zip_metric=None,
        by_quantile=False,
        verbose=False,
    ):
        self.cmap = cmap
        self.fitted_only = fitted_only
        self.inv_lambda = 1.0 / (amplitude_scaling_std**2)
        self.scale_clip_low = 1.0 / amplitude_scaling_limit
        self.scale_clip_high = amplitude_scaling_limit
        self.dist_vmax = dist_vmax
        self.show_values = show_values
        self.title = "grid mean dists"
        self.n_clust = n_clust
        self.n_iter = n_iter
        self.common_chans = common_chans
        self.inherit_chans = inherit_chans
        self.impute_before_center = impute_before_center
        self.min_overlap = min_overlap
        self.merge_on_waveform_radius = merge_on_waveform_radius
        self.with_proportions = with_proportions
        self.zip_metric = zip_metric
        self.verbose = verbose
        self.by_quantile = by_quantile

    def draw(self, panel, gmm, unit_id, impose_labels=None):
        n_clust = self.n_clust or gmm.kmeans_nclust
        n_iter = self.n_iter or gmm.kmeans_niter
        in_unit, labels, weights = gmm.kmeanspp(
            unit_id,
            n_clust=n_clust,
            n_iter=n_iter,
            with_proportions=self.with_proportions,
            verbose=self.verbose,
        )
        if impose_labels is not None:
            labels = impose_labels
            weights = torch.ones_like(weights)

        times = gmm.data.times_seconds[in_unit].numpy(force=True)
        amps = np.nanmax(gmm.data.amp_vecs[in_unit], 1)
        labels = labels.numpy(force=True)
        ids = np.unique(labels)
        ids = ids[ids >= 0]
        if self.verbose: print(f"gmm.KMeansPPSPlitPlot {ids=} {np.unique(labels, return_counts=True)=}")

        zip_metric = self.zip_metric or gmm.zip_metric
        zip_threshold = gmm.zip_threshold
        assert in_unit.shape == labels.shape
        in_unit, utd = gmm.get_training_data(unit_id, in_unit=in_unit, waveform_kind="reassign")
        assert in_unit.shape == labels.shape
        spike_ix, overlaps, badnesses = gmm[unit_id].spike_badnesses(
            utd["times"],
            utd["waveforms"],
            utd["waveform_channels"],
            kinds=(zip_metric, gmm.reassign_metric)
        )
        if gmm.zip_by_quantile:
            zip_threshold = torch.quantile(badnesses[zip_metric], zip_threshold)

        if ids.size > 1:
            top, bottom, below = panel.subfigures(nrows=3)
            ax_top, ax_top2 = top.subplots(ncols=2, width_ratios=[3, 1])
            axes = bottom.subplot_mosaic("dfe")
        else:
            ax_top = panel.subplots()

        ax_top.scatter(
            times,
            amps,
            c=glasbey1024[labels],
            s=4,
            lw=0,
        )
        if ids.size <= 1:
            return

        bads = badnesses[gmm.reassign_metric].numpy(force=True)
        spike_ix = spike_ix.numpy(force=True)
        assert in_unit[spike_ix].shape == bads.shape == labels[spike_ix].shape
        ax_top2.hist(
            [bads[labels[spike_ix] == l] for l in ids],
            color=glasbey1024[ids],
            histtype="step",
        )
        if self.verbose:
            print(f"assigning self.bads {bads.min()=} {bads.mean()=} {bads.max()=}")
            self.bads = bads

        new_units = []
        chans_kw = {} 
        if self.inherit_chans:
            chans_kw = dict(
                channels=gmm[unit_id].channels,
                max_channel=gmm[unit_id].max_channel,
            )
        if self.verbose: print(f"{self.impute_before_center=} {self.common_chans=}")
        chans_subunit = []
        for j, label in enumerate(ids):
            u = spike_interp.InterpUnit(
                do_interp=False,
                **gmm.unit_kw | dict(
                    impute_before_center=self.impute_before_center,
                    channel_strategy_snr_min=gmm[unit_id].channel_strategy_snr_min / 2,
                ),
            )
            inu = in_unit[np.flatnonzero(labels == label)]
            w = None if weights is None else weights[labels == label, j]
            inu, train_data = gmm.get_training_data(
                unit_id,
                waveform_kind="original",
                in_unit=inu,
                sampling_method=gmm.sampling_method,
            )
            chans_subunit.append(train_data['waveform_channels'])
            try:
                u.fit_center(
                    **train_data,
                    padded_geom=gmm.data.padded_registered_geom,
                    show_progress=False,
                    weights=w,
                    **chans_kw,
                )
                if gmm.cov_kind == "global":
                    if self.verbose: print(f"overwrite {u.var.min()=} {u.var.max()=} {gmm.var=}")
                    u.var.fill_(gmm.var)
                new_units.append(u)
            except ValueError:
                continue
        ju = [(j, u) for j, u in enumerate(new_units) if u.n_chans_unit]
        if self.verbose:
            print("assigning self.ju,in_unit,split_labels")
            self.ju = ju
            self.in_unit = in_unit
            self.split_labels = labels

        # plot new unit maxchan wfs and old one in black
        gmc = gmm[unit_id].max_channel
        for ax, pick in zip((axes["d"], axes["f"]), ("unit", "shared")):
            ax.axhline(0, c="k", lw=0.8)
            all_means = []
            for j, unit in ju:
                if unit.do_interp:
                    times = unit.interp.grid.squeeze()
                    if self.fitted_only:
                        times = times[unit.interp.grid_fitted]
                else:
                    times = torch.tensor([sum(gmm.t_bounds) / 2]).to(gmm.device)
                times = torch.atleast_1d(times)

                chans = torch.full((times.numel(),), unit.max_channel if pick == "unit" else gmc, device=times.device)
                means = unit.get_means(times).to(gmm.device)
                if j > 0:
                    all_means.append(means.mean(0))
                means = unit.to_waveform_channels(means, waveform_channels=chans[:, None])
                means = means[..., 0]
                means = gmm.data.tpca._inverse_transform_in_probe(means)
                means = means.numpy(force=True)
                color = glasbey1024[j]

                lines = np.stack(
                    (np.broadcast_to(np.arange(means.shape[1])[None], means.shape), means),
                    axis=-1,
                )
                ax.add_collection(LineCollection(lines, colors=color, lw=1))
            ax.autoscale_view()
            ax.set_xticks([])
            ax.spines[["top", "right", "bottom"]].set_visible(False)
            ax.set_title(pick)

        # plot distance matrix
        kind = gmm.zip_metric
        min_overlap = self.min_overlap
        if self.min_overlap is None:
            min_overlap = gmm.min_overlap
        subset_channel_index = None
        if self.merge_on_waveform_radius:
            subset_channel_index = gmm.data.registered_reassign_channel_index
        nu = len(new_units)
        divergences = torch.full((nu, nu), torch.nan)
        if self.verbose: print(f"{self.inherit_chans=} {zip_metric=} {subset_channel_index is None=}")
        for i, ua in ju:
            # print(f"{i=} {ua.n_chans_unit=} {ua.channels.tolist()=}")
            for j, ub in ju:
                # print(f"{j=} {ub.n_chans_unit=} {ub.channels.tolist()=}")
                if i == j:
                    divergences[i, j] = 0
                    continue
                divergences[i, j] = ua.divergence(
                    ub,
                    kind=kind,
                    min_overlap=min_overlap,
                    subset_channel_index=subset_channel_index,
                    common_chans=self.common_chans,
                )
        dists = divergences.numpy(force=True)

        axis = axes["e"]
        im = axis.imshow(
            dists,
            vmin=0,
            vmax=self.dist_vmax,
            cmap=self.cmap,
            origin="lower",
            interpolation="none",
        )
        if self.show_values:
            for (j, i), label in np.ndenumerate(dists):
                axis.text(
                    i,
                    j,
                    f"{label:.2f}".lstrip("0"),
                    ha="center",
                    va="center",
                    clip_on=True,
                    fontsize=5,
                )
        panel.colorbar(im, ax=axis, shrink=0.3)
        axis.set_xticks(range(len(new_units)))
        axis.set_yticks(range(len(new_units)))
        for i, (tx, ty) in enumerate(
            zip(axis.xaxis.get_ticklabels(), axis.yaxis.get_ticklabels())
        ):
            tx.set_color(glasbey1024[i])
            ty.set_color(glasbey1024[i])
        axis.set_title(f"{kind} {zip_threshold}")

        dbads, bimodalities, reas_labels = gmm.bimodalities_prefit(
            new_units,
            pairs_valid=dists < zip_threshold,
            which_spikes=in_unit,
            common_chans=self.common_chans,
            impute_missing=self.impute_before_center,
        )
        # f0 = below
        f0, f1 = below.subfigures(ncols=2, width_ratios=(2, 1))
        # ax1 = f1.subplots()
        # im = ax1.imshow(
        #     bimodalities,
        #     vmin=0,
        #     vmax=self.dist_vmax,
        #     cmap=self.cmap,
        #     origin="lower",
        #     interpolation="none",
        # )
        # if self.show_values:
        #     for (j, i), label in np.ndenumerate(bimodalities):
        #         ax1.text(
        #             i,
        #             j,
        #             f"{label:.2f}".lstrip("0"),
        #             ha="center",
        #             va="center",
        #             clip_on=True,
        #             fontsize=5,
        #         )
        # panel.colorbar(im, ax=ax1, shrink=0.3)
        # ax1.set_xticks(range(len(new_units)))
        # ax1.set_yticks(range(len(new_units)))
        # for i, (tx, ty) in enumerate(
        #     zip(ax1.xaxis.get_ticklabels(), ax1.yaxis.get_ticklabels())
        # ):
        #     tx.set_color(glasbey1024[i])
        #     ty.set_color(glasbey1024[i])

        if len(dbads) > 1:
            axes0 = f0.subplots(nrows=len(dbads) - 1, ncols=len(dbads) - 1, sharex=True, sharey=True, squeeze=False)
            for i in range(1, len(dbads)):
                for j in range(len(dbads) - 1):
                    ax = axes0[i - 1, j]
                    if j >= i:
                        ax.set_visible(False)
                        continue
    
                    ax.scatter(
                        dbads[i],
                        dbads[j],
                        c=np.stack([glasbey1024[i], glasbey1024[j]], axis=0)[(dbads[i] < dbads[j]).astype(int)],
                        s=3,
                        lw=0,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"{bimodalities[i, j]:0.2f}", fontsize=5)
    
            # ax0.hist(dbads.todense(), bins=32, density=True, histtype="step")
            ax1 = f1.subplots()
            # chans histogram...
            mn = np.inf
            mx = -np.inf
            cs = []
            for j, ucc in enumerate(chans_subunit):
                ucc = ucc[ucc < gmm.data.n_chans_full].numpy(force=True)
                chans_subunit[j] = ucc
                mn = min(ucc.min(), mn)
                mx = max(ucc.max(), mx)
                cs.append(glasbey1024[j])
            ax1.hist(chans_subunit, histtype="bar", bins=np.arange(mn, mx + 1), color=cs, stacked=True)


class HDBScanSplitPlot(GMMPlot):
    kind = "split"
    width = 2
    height = 2

    def __init__(self, spike_kind="split"):
        self.spike_kind = spike_kind

    def draw(self, panel, gmm, unit_id):
        if self.spike_kind == "residual_full":
            (in_unit,) = (gmm.labels == unit_id).nonzero(as_tuple=True)
            features = torch.empty(
                (in_unit.numel(), gmm.residual_pca_rank), device=gmm.device
            )
            for sl, data in gmm.batches(in_unit):
                gmm[unit_id].residual_embed(**data, out=features[sl])
            z = features[:, : gmm.dpc_split_kw.rank].numpy(force=True)
        elif self.spike_kind == "split":
            _, in_unit, z = gmm.split_features(unit_id)
        elif self.spike_kind == "global":
            in_unit, data = gmm.get_training_data(unit_id)
            waveforms = gmm[unit_id].to_unit_channels(
                waveforms=data["waveforms"],
                times=data["times"],
                waveform_channels=data["waveform_channels"],
            )
            loadings, mean, components, svs = spike_interp.fit_pcas(
                waveforms.reshape(in_unit.numel(), -1),
                missing=None,
                empty=None,
                rank=gmm.dpc_split_kw.rank,
                show_progress=False,
            )
            z = loadings.numpy(force=True)
        else:
            assert False
        in_unit = in_unit.numpy(force=True)
        zu, idx, inv = np.unique(z, return_index=True, return_inverse=True, axis=0)

        import hdbscan

        clus = hdbscan.HDBSCAN(min_cluster_size=25, cluster_selection_epsilon=0.0)
        clus.fit(zu)
        labs = clus.labels_
        ax = panel.subplots()
        ax.scatter(*zu[labs < 0].T, color="gray", s=3, lw=0)
        ax.scatter(*zu[labs >= 0].T, c=glasbey1024[labs[labs >= 0]], s=3, lw=0)
        ax.set_title(self.spike_kind + " hdb")


class EmbedsOverTimePlot(GMMPlot):
    kind = "embeds"
    width = 4
    height = 3

    def __init__(self, colors="gbr"):
        self.colors = colors

    def draw(self, panel, gmm, unit_id):
        _, in_unit, z = gmm.split_features(unit_id)
        ax = panel.subplots()
        t = gmm.data.times_seconds[in_unit].numpy(force=True)
        for j, zz in enumerate(z.T):
            ax.scatter(t, zz, color=self.colors[j], s=3, lw=0)
        ax.set_ylabel("residual embed")


class ChansHeatmap(GMMPlot):
    kind = "tall"
    width = 3
    height = 3

    def __init__(self, cmap=plt.cm.magma):
        self.cmap = cmap

    def draw(self, panel, gmm, unit_id):
        (in_unit_full,) = torch.nonzero(gmm.labels == unit_id, as_tuple=True)
        spike_chans = gmm.data.original_static_channels[in_unit_full].numpy(force=True)
        ixs = spike_chans[spike_chans < gmm.data.n_chans_full]
        unique_ixs, counts = np.unique(ixs, return_counts=True)
        ax = panel.subplots()
        xy = gmm.data.registered_geom.numpy(force=True)
        s = ax.scatter(*xy[unique_ixs].T, c=counts, lw=0, cmap=self.cmap)
        plt.colorbar(s, ax=ax, shrink=0.3)
        ax.scatter(
            *xy[gmm[unit_id].channels_valid.numpy(force=True)].T,
            color="r",
            lw=1,
            fc="none",
        )
        ax.scatter(
            *xy[np.atleast_1d(gmm[unit_id].max_channel.numpy(force=True))].T,
            color="g",
            lw=0,
        )


class AmplitudesOverTimePlot(GMMPlot):
    kind = "embeds"
    width = 4
    height = 3
    
    def __init__(self, kinds=("recon", "maxchan_energy", 'model'), colors="bkrg"):
        self.kinds = kinds
        self.colors = dict(zip(kinds, colors))

    def draw(self, panel, gmm, unit_id):
        in_unit, utd = gmm.get_training_data(unit_id, waveform_kind="reassign")
        
        show = {}
        if "feat" in self.kinds:
            amps = utd["static_amp_vecs"].numpy(force=True)
            show['feat'] = np.nanmax(amps, axis=1)
        amps2 = utd["waveforms"]
        if 'recon' in self.kinds:
            n, r, c = amps2.shape
            amps2 = gmm.data.tpca._inverse_transform_in_probe(
                amps2.permute(0, 2, 1).reshape(n * c, r)
            )
            amps2 = amps2.reshape(n, -1, c).permute(0, 2, 1)
            show['recon'] = np.ptp(np.nan_to_num(amps2.numpy(force=True)), axis=(1, 2))
        if 'model' in self.kinds:
            recons = gmm[unit_id].get_means(utd["times"])
            recons = gmm[unit_id].to_waveform_channels(
                recons, waveform_channels=utd["waveform_channels"]
            )
            n, r, c = recons.shape
            recons = gmm.data.tpca._inverse_transform_in_probe(
                recons.permute(0, 2, 1).reshape(n * c, r)
            )
            recons = recons.reshape(n, -1, c).permute(0, 2, 1)
            show['model'] = np.ptp(np.nan_to_num(recons.numpy(force=True)), axis=(1, 2))
        if 'maxchan_energy' in self.kinds:
            wfs = torch.nan_to_num(utd['waveforms'])
            wfs = torch.linalg.norm(wfs, dim=1)
            wfs = wfs.max(dim=1).values
            show['maxchan_energy'] = wfs.numpy(force=True)

        ax = panel.subplots()
        t = utd["times"].numpy(force=True)
        for kind, a in show.items():
            ax.scatter(t, a, c=self.colors[kind], s=3, lw=0, label=kind)
        ax.legend(loc="upper left")
        ax.set_ylabel("amplitude")


class ISIHistogram(GMMPlot):
    kind = "small"
    width = 2
    height = 2

    def __init__(self, bin_ms=0.1, max_ms=5):
        super().__init__()
        self.bin_ms = bin_ms
        self.max_ms = max_ms

    def draw(self, panel, gmm, unit_id):
        axis = panel.subplots()
        times_s = gmm.data.times_seconds[gmm.labels == unit_id].numpy(force=True)
        dt_ms = np.diff(times_s) * 1000
        bin_edges = np.arange(
            0,
            self.max_ms + self.bin_ms,
            self.bin_ms,
        )
        counts, _ = np.histogram(dt_ms, bin_edges)
        # bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        # axis.bar(bin_centers, counts)
        axis.stairs(counts, bin_edges, color="k", fill=True)
        axis.set_xlabel("isi (ms)")
        axis.set_ylabel(f"count (out of {dt_ms.size} total isis)")


class BadnessesOverTimePlot(GMMPlot):
    kind = "embeds"
    width = 4
    height = 2

    def __init__(self, colors="rgb", kinds=None):
        self.colors = colors
        self.kinds = kinds

    def draw(self, panel, gmm, unit_id):
        in_unit, utd = gmm.get_training_data(unit_id, waveform_kind="reassign")
        kinds = self.kinds
        if self.kinds is None:
            kinds = (gmm.reassign_metric,)

        spike_ix, overlaps, badnesses = gmm[unit_id].spike_badnesses(
            utd["times"],
            utd["waveforms"],
            utd["waveform_channels"],
            kinds=kinds,
        )
        if not spike_ix.numel():
            return
        overlaps = overlaps.numpy(force=True)
        times = utd["times"][spike_ix].numpy(force=True)

        ax, ay = panel.subplots(ncols=2, width_ratios=[2, 1])
        ay.grid(True)
        for j, (kind, b) in enumerate(badnesses.items()):
            b = b.numpy(force=True)
            c = self.colors[j]
            ax.scatter(
                times,
                b,
                alpha=overlaps,
                s=3,
                c=c,
                lw=0,
                label=kind,
            )
            ay.ecdf(b, lw=1, color=c)
            ay.axvline(np.mean(b), color=c, lw=1)
            ay.axvline(np.median(b), color=c, ls="--", lw=1)
            ay.set_xlabel(kind)
            ay.set_ylabel("cdf")
        ax.legend(loc="upper left")
        ax.set_ylabel("badness")
        ay.set_yticks([0, 0.25, 0.5, 0.75, 1])


class FeaturesVsBadnessesPlot(GMMPlot):
    kind = "embeds"
    width = 4
    height = 2

    def __init__(self, colors="rgb", kinds=("1-r^2", "1-scaledr^2")):
        self.colors = colors
        self.kinds = kinds

    def draw(self, panel, gmm, unit_id):
        # residual embed feats
        if gmm.dpc_split_kw.split_on_train:
            _, in_unit_, z = gmm.split_features(unit_id)
            in_unit, utd = gmm.get_training_data(unit_id, in_unit=in_unit_)
        else:
            in_unit, utd = gmm.get_training_data(unit_id)
            _, in_unit_, z = gmm.split_features(unit_id, in_unit)
        assert torch.equal(in_unit_, in_unit)
        overlaps, rel_ix = gmm[unit_id].overlaps(utd["waveform_channels"])

        # badnesses
        spike_ix, overlaps, badnesses = gmm[unit_id].spike_badnesses(
            utd["times"],
            utd["waveforms"],
            utd["waveform_channels"],
            kinds=self.kinds,
            overlaps=overlaps,
            rel_ix=rel_ix,
        )
        if not spike_ix.size:
            return
        overlaps = overlaps.numpy(force=True)
        badnesses = {k: v.numpy(force=True) for k, v in badnesses.items()}

        # amplitudes
        amps = torch.nan_to_num(torch.linalg.norm(utd['waveforms'], dim=1)).max(dim=1).values.numpy(force=True)

        axes = panel.subplots(ncols=z.shape[1] + 1, sharey=True)
        for ax, feat, featname in zip(
            axes,
            (amps, *z.T),
            ("amps", *[f"emb{x}" for x in range(z.shape[1])]),
        ):
            for j, kind in enumerate(self.kinds):
                ax.scatter(feat, badnesses[kind], color=self.colors[j], s=3, lw=0)
            ax.set_xlabel(featname)
        axes[0].set_ylabel("badnesses")


class GridMeansMultiChanPlot(GMMPlot):
    kind = "waveform"
    width = 5
    height = 5

    def __init__(self, cmap=plt.cm.rainbow, time_range=None, fitted_only=True):
        self.cmap = cmap
        self.time_range = time_range
        self.fitted_only = fitted_only

    def draw(self, panel, gmm, unit_id):
        times, chans, recons = get_means(gmm, unit_id, fitted_only=self.fitted_only, time_range=self.time_range)
        times = times.numpy(force=True)
        chans = chans.numpy(force=True)
        if gmm[unit_id].do_interp:
            c = times
            c = (c - self.time_range[0]) / (self.time_range[1] - self.time_range[0])
            colors = self.cmap(c)
        else:
            colors = "r"
        recons = recons.numpy(force=True)
        maa = np.nanmax(np.abs(recons))

        ax = panel.subplots()
        geomplot(
            recons,
            max_channels=chans,
            channel_index=gmm.data.registered_reassign_channel_index.numpy(force=True),
            geom=gmm.data.registered_geom.numpy(force=True),
            max_abs_amp=maa,
            lw=1,
            colors=colors,
            show_zero=False,
            subar=True,
            msbar=False,
            zlim="tight",
        )
        ax.axis("off")
        ax.set_title("mean")


class GridMeansSingleChanPlot(GMMPlot):
    kind = "small"
    width = 3
    height = 2

    def __init__(self, cmap=plt.cm.rainbow, time_range=None, fitted_only=True):
        self.cmap = cmap
        self.time_range = time_range
        self.fitted_only = fitted_only

    def draw(self, panel, gmm, unit_id):
        # amps = np.nanmax(amps, axis=1)
        times, chans, recons = get_means(gmm, unit_id, single_chan=True, fitted_only=self.fitted_only, time_range=self.time_range)
        if gmm[unit_id].do_interp:
            c = times.numpy(force=True)
            c = (c - time_range[0]) / (time_range[1] - time_range[0])
            colors = self.cmap(c)
        else:
            colors = "r"

        recons = recons.numpy(force=True)

        ax = panel.subplots()
        ax.axhline(0, c="k", lw=0.8)
        lines = np.stack(
            (np.broadcast_to(np.arange(recons.shape[1])[None], recons.shape), recons),
            axis=-1,
        )
        ax.add_collection(LineCollection(lines, colors=colors, lw=1))
        ax.autoscale_view()
        ax.set_xticks([])
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.set_title("mean")


def get_means(gmm, unit_id, main_channel=None, single_chan=False, fitted_only=True, time_range=None):
    if gmm[unit_id].do_interp:
        times = gmm[unit_id].interp.grid.squeeze()
        if fitted_only:
            times = times[gmm[unit_id].interp.grid_fitted]
    else:
        times = torch.tensor([sum(gmm.t_bounds) / 2]).to(gmm.device)
    times = torch.atleast_1d(times)
    if time_range is None:
        time_range = times.min(), times.max()
    if main_channel is None:
        main_channel = gmm[unit_id].max_channel
    chans = torch.full(
        (times.numel(),), main_channel, device=times.device
    )
    recons = gmm[unit_id].get_means(times).to(gmm.device)
    if single_chan:
        recons = gmm[unit_id].to_waveform_channels(
            recons, waveform_channels=chans[:, None]
        )
        recons = recons[..., 0]
        recons = gmm.data.tpca._inverse_transform_in_probe(recons)
    else:
        waveform_channels = gmm.data.registered_reassign_channel_index[chans]
        if waveform_channels.ndim == 1:
            waveform_channels = waveform_channels[None]
        recons = gmm[unit_id].get_means(times)
        recons = gmm[unit_id].to_waveform_channels(
            recons, waveform_channels=waveform_channels
        )
        recons = gmm.data.tpca.inverse_transform(
            recons, chans, channel_index=gmm.data.registered_reassign_channel_index
        )

    return times, chans, recons


class InputWaveformsMultiChanPlot(GMMPlot):
    kind = "waveform"
    width = 5
    height = 5

    def __init__(self, cmap=plt.cm.rainbow, max_plot=250, rg=0, time_range=None, imputation_kind=None):
        self.cmap = cmap
        self.max_plot = max_plot
        self.rg = np.random.default_rng(0)
        self.time_range = time_range
        self.imputation_kind = imputation_kind

    def draw(self, panel, gmm, unit_id):
        in_unit, utd = gmm.get_training_data(unit_id)
        wh = slice(None)
        if len(in_unit) > self.max_plot:
            wh = np.sort(self.rg.choice(len(in_unit), self.max_plot, replace=False))
            wh = torch.tensor(wh).to(in_unit)
        times = utd["times"].squeeze()[wh]
        time_range = self.time_range
        if time_range is None:
            time_range = times.min(), times.max()
        waveform_channels = utd["waveform_channels"][wh]
        waveforms = utd["waveforms"][wh]
        n, r, c = waveforms.shape
        if self.imputation_kind:
            waveforms = gmm[unit_id].impute(
                times,
                waveforms,
                waveform_channels,
                waveform_channel_index=utd["waveform_channel_index"],
                imputation_kind=self.imputation_kind,
                padded_registered_geom=gmm.data.padded_registered_geom,
            )
            waveforms = waveforms.reshape(n, r, gmm[unit_id].n_chans_unit)
            waveform_channels = gmm[unit_id].channels.numpy(force=True)
            waveform_channels = np.broadcast_to(waveform_channels[None], (n, *waveform_channels.shape))
        else:
            waveform_channels = waveform_channels.numpy(force=True)
        n, r, c = waveforms.shape

        waveforms = gmm.data.tpca._inverse_transform_in_probe(
            waveforms.permute(0, 2, 1).reshape(n * c, r)
        )
        waveforms = waveforms.reshape(n, c, -1).permute(0, 2, 1)
        waveforms = waveforms.numpy(force=True)
        c = times.numpy(force=True)
        c = (c - time_range[0]) / (time_range[1] - time_range[0])
        colors = self.cmap(c)
        maa = np.nanmax(np.abs(waveforms))

        ax = panel.subplots()
        geomplot(
            waveforms,
            channels=waveform_channels,
            geom=gmm.data.registered_geom.numpy(force=True),
            max_abs_amp=maa,
            lw=1,
            colors=colors,
            show_zero=False,
            subar=True,
            msbar=False,
            zlim="tight",
        )
        ax.axis("off")
        ax.set_title("input waveforms")


class InputWaveformsSingleChanPlot(GMMPlot):
    kind = "small"
    width = 3
    height = 2

    def __init__(self, cmap=plt.cm.rainbow, max_plot=250, rg=0, time_range=None):
        self.cmap = cmap
        self.max_plot = max_plot
        self.rg = np.random.default_rng(0)
        self.time_range = time_range

    def draw(self, panel, gmm, unit_id):
        in_unit, utd = gmm.get_training_data(unit_id, waveform_kind="reassign")
        wh = slice(None)
        if len(in_unit) > self.max_plot:
            wh = np.sort(self.rg.choice(len(in_unit), self.max_plot, replace=False))
            wh = torch.tensor(wh).to(in_unit)
        times = utd["times"].squeeze()[wh]
        time_range = self.time_range
        if time_range is None:
            time_range = times.min(), times.max()
        waveforms = utd["waveforms"][wh]
        waveform_channels = utd["waveform_channels"][wh]

        main_channel = gmm[unit_id].max_channel.to(waveform_channels)
        nixs, cixs = torch.nonzero(waveform_channels == main_channel, as_tuple=True)
        waveforms = waveforms[nixs]
        times = times[nixs]
        waveforms = torch.take_along_dim(waveforms, cixs[:, None, None], dim=2)[:, :, 0]

        waveforms = gmm.data.tpca._inverse_transform_in_probe(waveforms)
        c = times.numpy(force=True)
        c = (c - time_range[0]) / (time_range[1] - time_range[0])
        colors = self.cmap(c)
        waveforms = waveforms.numpy(force=True)

        ax = panel.subplots()
        ax.axhline(0, c="k", lw=0.8)
        lines = np.stack(
            (
                np.broadcast_to(np.arange(waveforms.shape[1])[None], waveforms.shape),
                waveforms,
            ),
            axis=-1,
        )
        ax.add_collection(LineCollection(lines, colors=colors, lw=1))
        ax.autoscale_view()
        ax.set_xticks([])
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.set_title("input waveforms")


class InputWaveformsSingleChanOverTimePlot(GMMPlot):
    kind = "column"
    width = 2
    height = np.inf

    def __init__(
        self,
        cmap=plt.cm.rainbow,
        max_plot=100,
        rg=0,
        time_range=None,
        dt=300,
        max_bins=15,
        channel="unit",
    ):
        self.cmap = cmap
        self.max_plot = max_plot
        self.rg = np.random.default_rng(0)
        self.time_range = time_range
        self.dt = dt
        self.max_bins = max_bins
        self.channel = channel

    def draw(self, panel, gmm, unit_id):
        time_range = self.time_range
        if time_range is None:
            times = gmm.data.times_seconds[gmm.labels == unit_id]
            time_range = times.min().numpy(force=True), times.max().numpy(force=True)
        n_bins = int((time_range[1] - time_range[0]) // self.dt)
        n_bins = min(self.max_bins, max(1, n_bins))
        bin_edges = np.linspace(*time_range, num=n_bins)

        axes = panel.subplots(nrows=self.max_bins, sharex=True, sharey=True)
        # axes[0].set_title(f"input waveforms\non {self.channel} channel")
        axes[0].set_title(self.channel)

        for bin_left, bin_right, ax in zip(bin_edges, bin_edges[1:], axes.flat):
            times_valid = gmm.data.times_seconds == gmm.data.times_seconds.clip(
                bin_left, bin_right
            )
            in_unit = torch.logical_and(
                gmm.labels == unit_id,
                times_valid.to(gmm.labels),
            )
            (in_unit,) = torch.nonzero(in_unit, as_tuple=True)

            in_unit, utd = gmm.get_training_data(
                unit_id, n=self.max_plot, in_unit=in_unit, waveform_kind="reassign"
            )
            times = torch.atleast_1d(utd["times"].squeeze())
            waveforms = utd["waveforms"]
            waveform_channels = utd["waveform_channels"]

            if self.channel == "unit":
                main_channel = gmm[unit_id].max_channel.to(waveform_channels)
                nixs, cixs = torch.nonzero(
                    waveform_channels == main_channel, as_tuple=True
                )
            elif self.channel == "natural":
                chans = gmm.data.static_main_channels[in_unit].to(waveform_channels)
                nixs, cixs = torch.nonzero(
                    waveform_channels == chans[:, None], as_tuple=True
                )
            else:
                assert False
            waveforms = waveforms[nixs]
            times = times[nixs]
            waveforms = torch.take_along_dim(waveforms, cixs[:, None, None], dim=2)[
                :, :, 0
            ]

            waveforms = gmm.data.tpca._inverse_transform_in_probe(waveforms)
            c = times.numpy(force=True)
            c = (c - time_range[0]) / (time_range[1] - time_range[0])
            colors = self.cmap(c)
            waveforms = waveforms.numpy(force=True)

            ax.axhline(0, c="k", lw=0.8)
            lines = np.stack(
                (
                    np.broadcast_to(
                        np.arange(waveforms.shape[1])[None], waveforms.shape
                    ),
                    waveforms,
                ),
                axis=-1,
            )
            ax.add_collection(LineCollection(lines, colors=colors, lw=1))
            ax.autoscale_view()
            ax.set_xticks([])
            ax.spines[["top", "right", "bottom"]].set_visible(False)
            ax.set_ylabel(f"{bin_left:0.1f} - {bin_right:0.1f} s")
        for ax in axes.flat[n_bins:]:
            ax.axis("off")


class ResidualsSingleChanPlot(GMMPlot):
    kind = "small"
    width = 2
    height = 2

    def __init__(
        self,
        cmap=plt.cm.rainbow,
        max_plot=250,
        rg=0,
        time_range=None,
        scaled=True,
        amplitude_scaling_std=np.sqrt(0.001),
        amplitude_scaling_limit=1.2,
    ):
        self.cmap = cmap
        self.max_plot = max_plot
        self.rg = np.random.default_rng(0)
        self.time_range = time_range
        self.scaled = scaled
        self.inv_lambda = 1.0 / (amplitude_scaling_std**2)
        self.scale_clip_low = 1.0 / amplitude_scaling_limit
        self.scale_clip_high = amplitude_scaling_limit

    def draw(self, panel, gmm, unit_id):
        in_unit, utd = gmm.get_training_data(unit_id, waveform_kind="reassign")
        wh = slice(None)
        if len(in_unit) > self.max_plot:
            wh = np.sort(self.rg.choice(len(in_unit), self.max_plot, replace=False))
            wh = torch.tensor(wh).to(in_unit)

        times = utd["times"].squeeze()[wh]
        time_range = self.time_range
        if time_range is None:
            time_range = times.min(), times.max()
        chans = gmm.data.channels[in_unit[wh]]
        waveform_channels = utd["waveform_channels"][wh]

        waveforms = utd["waveforms"][wh]
        mask = torch.isfinite(waveforms[:, 0, :]).unsqueeze(1).to(waveforms)
        waveforms = torch.nan_to_num(waveforms)

        recons = gmm[unit_id].get_means(times)
        recons = gmm[unit_id].to_waveform_channels(
            recons, waveform_channels=waveform_channels
        )
        recons = torch.nan_to_num(recons * mask)

        scalings = torch.ones_like(recons)
        if self.scaled:
            dots = recons.mul(waveforms).sum(dim=(1, 2))
            recons_sumsq = recons.square().sum(dim=(1, 2))
            scalings = (dots + self.inv_lambda).div_(recons_sumsq + self.inv_lambda)
            scalings = scalings.clip_(self.scale_clip_low, self.scale_clip_high)[
                :, None, None
            ]
        waveforms = waveforms - scalings * recons

        main_channel = gmm[unit_id].max_channel.to(waveform_channels)
        nixs, cixs = torch.nonzero(waveform_channels == main_channel, as_tuple=True)
        waveforms = waveforms[nixs]
        times = times[nixs]
        waveforms = torch.take_along_dim(waveforms, cixs[:, None, None], dim=2)[:, :, 0]

        waveforms = gmm.data.tpca._inverse_transform_in_probe(waveforms)

        c = times.numpy(force=True)
        c = (c - time_range[0]) / (time_range[1] - time_range[0])
        colors = self.cmap(c)
        waveforms = waveforms.numpy(force=True)
        chans = chans.numpy(force=True)

        ax = panel.subplots()
        ax.axhline(0, c="k", lw=0.8)
        lines = np.stack(
            (
                np.broadcast_to(np.arange(waveforms.shape[1])[None], waveforms.shape),
                waveforms,
            ),
            axis=-1,
        )
        ax.add_collection(LineCollection(lines, colors=colors, lw=1))
        ax.autoscale_view()
        ax.set_xticks([])
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        if self.scaled:
            ax.set_title("scaled residuals")
        else:
            ax.set_title("residuals")


class GridMeanDistancesPlot(GMMPlot):
    kind = "dists"
    width = 5
    height = 5

    def __init__(
        self,
        cmap=plt.cm.rainbow,
        fitted_only=True,
        scaled=True,
        amplitude_scaling_std=np.sqrt(0.001),
        amplitude_scaling_limit=1.2,
        dist_vmax=1.0,
        show_values=True,
    ):
        self.cmap = cmap
        self.fitted_only = fitted_only
        self.scaled = scaled
        self.inv_lambda = 1.0 / (amplitude_scaling_std**2)
        self.scale_clip_low = 1.0 / amplitude_scaling_limit
        self.scale_clip_high = amplitude_scaling_limit
        self.dist_vmax = dist_vmax
        self.show_values = show_values
        self.title = "grid mean dists"
        if self.scaled:
            self.title = f"scaled {self.title}"

    def draw(self, panel, gmm, unit_id):
        # amps = np.nanmax(amps, axis=1)
        if gmm[unit_id].do_interp:
            times = gmm[unit_id].interp.grid.squeeze()
            if self.fitted_only:
                times = times[gmm[unit_id].interp.grid_fitted]
            means = gmm[unit_id].get_means(times).reshape(len(times), -1)
            l2s = means.square().sum(1)

            if self.scaled:
                dots = (means[:, None, :] * means[None, :, :]).sum(2)
                scalings = (dots + self.inv_lambda).div_(l2s + self.inv_lambda)
                scalings = scalings.clip_(self.scale_clip_low, self.scale_clip_high)
            else:
                scalings = torch.ones_like(l2s[:, None] + l2s[None, :])

            dists = (
                means[:, None]
                .sub(scalings[:, :, None] * means[None])
                .square()
                .sum(2)
                .div(l2s)
            )
            dists = dists.numpy(force=True)
            times = times.numpy(force=True)
        else:
            dists = [[0]]
            times = [0]

        axis = panel.subplots()
        im = axis.imshow(
            dists,
            vmin=0,
            vmax=self.dist_vmax,
            cmap=self.cmap,
            origin="lower",
            interpolation="none",
        )
        if self.show_values:
            for (j, i), label in np.ndenumerate(dists):
                axis.text(i, j, f"{label:.2f}", ha="center", va="center", clip_on=True)
        panel.colorbar(im, ax=axis, shrink=0.3)
        axis.set_xticks(range(len(times)), [f"{t:0.1f}" for t in times])
        axis.set_yticks(range(len(times)), [f"{t:0.1f}" for t in times])
        axis.set_title(self.title)


# -- merge plots

class GMMMergePlot(GMMPlot):

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def get_neighbors(self, gmm, unit_id, reversed=False):
        unit_dists = gmm.central_divergences(units_a=torch.tensor([unit_id]))[0]
        unit_ids = gmm.unit_ids()
        neighbors = torch.argsort((unit_ids != unit_id).to(unit_dists) + unit_dists)
        assert unit_ids[neighbors[0]] == unit_id
        neighbors = neighbors[: self.n_neighbors + 1]
        neighbors = neighbors[torch.isfinite(unit_dists[neighbors])]
        if reversed:
            neighbors = torch.flip(neighbors, (0,))
        return neighbors.numpy(force=True)


class NearbyTimesVAmps(GMMMergePlot):
    kind = "wall"
    width = 3
    height = 1.5

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def draw(self, panel, gmm, unit_id):
        neighbors = self.get_neighbors(gmm, unit_id, reversed=True)
        ax = panel.subplots()
        for u in neighbors:
            (inu,) = torch.nonzero(gmm.labels == u, as_tuple=True)
            t = gmm.data.times_seconds[inu].numpy(force=True)
            a = gmm.data.amps[inu]
            ax.scatter(t, a, color=glasbey1024[u % len(glasbey1024)], s=3, lw=0)
        ax.set_ylabel("amp")


class ViolatorTimesVAmps(GMMPlot):
    kind = "wide"
    width = 3
    height = 1.5

    def __init__(self, n_neighbors=5, viol_ms=1.0):
        self.n_neighbors = n_neighbors
        self.viol_ms = viol_ms

    def draw(self, panel, gmm, unit_id):
        ax = panel.subplots()
        (inu,) = torch.nonzero(gmm.labels == unit_id, as_tuple=True)
        t = gmm.data.times_seconds[inu].numpy(force=True)
        a = gmm.data.amps[inu]

        dt_ms = np.diff(t) * 1000
        small = dt_ms <= self.viol_ms
        small = np.logical_or(
            np.pad(small, (1, 0), constant_values=False),
            np.pad(small, (0, 1), constant_values=False),
        )
        big = np.logical_not(small)
        ax.scatter(t[big], a[big], c="k", s=5, lw=0)
        ax.scatter(t[small], a[small], c="r", s=5, lw=0)
        ax.set_ylabel("amp")


class ViolatorTimesVBadness(GMMPlot):
    kind = "vwide"
    width = 3
    height = 1.5

    def __init__(self, n_neighbors=5, viol_ms=1.0, kind=None):
        self.n_neighbors = n_neighbors
        self.viol_ms = viol_ms
        self.kind = kind

    def draw(self, panel, gmm, unit_id):
        ax = panel.subplots()
        (inu,) = torch.nonzero(gmm.labels == unit_id, as_tuple=True)
        t = gmm.data.times_seconds[inu].numpy(force=True)

        badness = gmm.reassignment_divergences(
            which_spikes=inu,
            unit_ids=[unit_id],
            show_progress=False,
            kind=self.kind,
        )
        a = np.full_like(t, np.inf)
        a[badness.coords[1]] = badness.data
        fin = np.isfinite(a)
        inf = np.logical_not(fin)

        dt_ms = np.diff(t) * 1000
        small = dt_ms <= self.viol_ms
        small = np.logical_or(
            np.pad(small, (1, 0), constant_values=False),
            np.pad(small, (0, 1), constant_values=False),
        )
        big = np.logical_not(small)
        ax.scatter(t[big & fin], a[big & fin], c="k", s=5, lw=0)
        ax.scatter(t[small & fin], a[small & fin], c="r", s=5, lw=0)
        ax.scatter(t[big & inf], np.ones_like(a[big & inf]), c="k", s=5, lw=0, marker="x")
        ax.scatter(t[small & inf], np.ones_like(a[small & inf]), c="r", s=5, lw=0, marker="x")
        ax.set_ylabel(gmm.reassign_metric)


class NearbyTimesVBadness(GMMMergePlot):
    kind = "vwall"
    width = 3
    height = 1.5

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def draw(self, panel, gmm, unit_id):
        neighbors = self.get_neighbors(gmm, unit_id, reversed=True)
        ax = panel.subplots()
        for u in neighbors:
            (inu,) = torch.nonzero(gmm.labels == u, as_tuple=True)
            t = gmm.data.times_seconds[inu].numpy(force=True)

            badness = gmm.reassignment_divergences(
                which_spikes=inu,
                unit_ids=[unit_id],
                show_progress=False,
            )
            a = np.full_like(t, np.inf)
            a[badness.coords[1]] = badness.data
            a = np.nan_to_num(a, nan=1.0, posinf=1.0, copy=False)
            ax.scatter(t, a, color=glasbey1024[u % len(glasbey1024)], s=3, lw=0)
        ax.set_ylabel("badness")


class NearbyMeansSingleChan(GMMMergePlot):
    kind = "small"
    width = 2
    height = 2

    def __init__(self, fitted_only=True, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.fitted_only = fitted_only

    def draw(self, panel, gmm, unit_id):
        neighbors = self.get_neighbors(gmm, unit_id, reversed=True)

        # get their means
        main_channel = gmm[unit_id].max_channel
        neighbor_means = []
        for u in neighbors:
            times, chans, means = get_means(gmm, u, main_channel=main_channel, single_chan=True, fitted_only=self.fitted_only)
            means = means.mean(0)
            neighbor_means.append(means.numpy(force=True))
        neighbor_means = np.stack(neighbor_means, axis=0)

        # process into a line collection
        t_coords = np.broadcast_to(np.arange(neighbor_means.shape[1])[None], neighbor_means.shape)
        lines = np.stack((t_coords, neighbor_means), axis=-1)
        colors = glasbey1024[neighbors % len(glasbey1024)]
        lines = LineCollection(lines, colors=colors, lw=1)

        # draw
        ax = panel.subplots()
        ax.axhline(0, c="k", lw=0.8)
        ax.add_collection(lines)
        ax.autoscale_view()
        ax.set_xticks([])
        ax.spines[["top", "right", "bottom"]].set_visible(False)


class NearbyMeansMultiChan(GMMMergePlot):
    kind = "tall"
    width = 3
    height = 4

    def __init__(self, fitted_only=True, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.fitted_only = fitted_only

    def draw(self, panel, gmm, unit_id):
        neighbors = self.get_neighbors(gmm, unit_id, reversed=True)

        # get their means
        main_channel = gmm[unit_id].max_channel
        neighbor_means = []
        max_chans = []
        for u in neighbors:
            times, chans, means = get_means(gmm, u, main_channel=main_channel, fitted_only=self.fitted_only)
            assert means.ndim == 3
            means = means.mean(0)
            neighbor_means.append(means.numpy(force=True))
            max_chans.append(chans.numpy(force=True))
        neighbor_means = np.stack(neighbor_means, axis=0)
        max_chans = np.concatenate(max_chans)
        colors = glasbey1024[neighbors % len(glasbey1024)]

        # draw
        ax = panel.subplots()
        geomplot(
            neighbor_means,
            max_channels=max_chans,
            channel_index=gmm.data.registered_reassign_channel_index.numpy(force=True),
            geom=gmm.data.registered_geom.numpy(force=True),
            lw=1,
            colors=colors,
            show_zero=False,
            subar=True,
            msbar=False,
            zlim="tight",
            ax=ax,
        )
        ax.axis("off")


class NeighborBimodality(GMMMergePlot):
    kind = "vtall"
    width = 3
    height = 10

    def __init__(self, n_neighbors=5, badness_kind=None, do_reg=False, masked=False, mask_radius_s=5.0, impute_missing=False, max_spikes=2048, cut=None, kind="isotonic", verbose=False):
        self.n_neighbors = n_neighbors
        self.badness_kind = badness_kind
        self.do_reg = do_reg
        self.masked = masked
        self.mask_radius_s = mask_radius_s
        self.impute_missing = impute_missing
        self.max_spikes = max_spikes
        self.cut = None
        self.kind = kind
        self.verbose = verbose

    def draw(self, panel, gmm, unit_id):
        kind = self.badness_kind
        if kind is None:
            kind = gmm.reassign_metric
        if self.verbose:
            print(f"{kind=} {self.masked=} {self.impute_missing=}")
        (in_self,) = torch.nonzero(gmm.labels == unit_id, as_tuple=True)
        neighbors = self.get_neighbors(gmm, unit_id)
        # remove self
        neighbors = neighbors[1:]
        axes = panel.subplots(nrows=self.n_neighbors, ncols=2 + self.do_reg, squeeze=False)
        for row in axes[len(neighbors):]:
            row[0].axis("off")
            row[1].axis("off")

        if self.masked:
            times_self = gmm.data.times_seconds[in_self][:, None].cpu()
            kdtree_self = KDTree(times_self.numpy(force=True))

        for u, row in zip(neighbors, axes):
            (inu,) = torch.nonzero(gmm.labels == u, as_tuple=True)

            if self.masked:
                times_u = gmm.data.times_seconds[inu][:, None].cpu()
                kdtree_u = KDTree(times_u.numpy(force=True))

                self_matched = np.isfinite(
                    kdtree_u.query(times_self, distance_upper_bound=self.mask_radius_s)[0]
                )
                u_matched = np.isfinite(
                    kdtree_self.query(times_u,  distance_upper_bound=self.mask_radius_s)[0]
                )
                in_self_local = in_self.numpy(force=True)[self_matched]
                inu_local = inu.numpy(force=True)[u_matched]
            else:
                in_self_local = in_self.numpy(force=True)
                inu_local = inu.numpy(force=True)

            rg = np.random.default_rng(0)
            if inu_local.size > self.max_spikes:
                inu_local = rg.choice(inu_local, size=self.max_spikes, replace=False)
                inu_local.sort()
            if in_self_local.size > self.max_spikes:
                in_self_local = rg.choice(in_self_local, size=self.max_spikes, replace=False)
                in_self_local.sort()

            nu = inu_local.size
            ns = in_self_local.size

            if min(nu, ns) < 2:
                continue

            which = np.concatenate((in_self_local, inu_local))
            order = np.argsort(which)
            which = which[order]
            identity = np.zeros_like(which)
            ntot = ns + nu
            identity[:ns] = unit_id
            identity[ns:] = u
            identity = identity[order]
            sample_weights = np.zeros(which.shape)
            sample_weights[:ns] = (nu / ntot) / 0.5
            sample_weights[ns:] = (ns / ntot) / 0.5
            sample_weights = sample_weights[order]

            badness = gmm.reassignment_divergences(
                which_spikes=torch.from_numpy(which).to(gmm.labels),
                unit_ids=[unit_id, u],
                show_progress=False,
                kind=kind,
                impute_missing=self.impute_missing,
            )
            a = np.full(badness.shape, np.inf)
            a[badness.coords] = badness.data
            a = np.nan_to_num(a, nan=gmm.match_threshold, posinf=gmm.match_threshold, copy=False)
            self_badness, u_badness = a
            pear = pearsonr(self_badness, u_badness)

            row[0].scatter(
                u_badness,
                self_badness,
                c=glasbey1024[identity % len(glasbey1024)],
                s=4,
                lw=0,
                alpha=0.5,
            )
            row[0].set_title(f"{ns=} {nu=} 1-rho={1.0 - pear.statistic:.2f} p={pear.pvalue:.2f}", fontsize=6)
            row[0].set_xlabel(f"{u}: {self.badness_kind}")
            row[0].set_ylabel(f"{unit_id}: {self.badness_kind}")

            # closer to me = self_badness < u_badness = u_badness - self_badness > 0 = to the right
            dbad = u_badness - self_badness
            unique_dbad, inverse, counts = np.unique(dbad, return_counts=True, return_inverse=True)
            # weights = counts.copy().astype(float)
            weights = np.zeros(counts.shape)
            np.add.at(weights, inverse, sample_weights)

            n, bins, patches = row[1].hist(dbad, bins=64, histtype="step", color="b", density=True)
            hist, _, _ = row[1].hist(unique_dbad, bins=bins, histtype="step", color="b", linestyle=":", density=True)
            histw, _, _ = row[1].hist(unique_dbad, weights=weights, bins=bins, histtype="step", color="r", density=True)
            bc = 0.5 * (bins[1:] + bins[:-1])

            ds_ud, x, m, m_ud, cut_ud = smoothed_dipscore_at(self.cut, unique_dbad, sample_weights=counts.astype(float), kind=self.kind)
            row[1].plot(x, m, color="g", lw=1)
            row[1].plot(x, m_ud, color="g", lw=1, ls=(0, (0.5, 0.5)))
            row[1].axvline(cut_ud, lw=0.8, color="g")
            ds_udw, xw, mw, m_udw, cut_udw = smoothed_dipscore_at(self.cut, unique_dbad, sample_weights=weights, kind=self.kind)
            row[1].plot(xw, mw, color="orange", lw=1)
            row[1].plot(xw, m_udw, color="orange", lw=1, ls=(0, (0.5, 0.5)))
            row[1].axvline(cut_udw, lw=0.8, color="orange")

            dbin = np.diff(bc).mean()
            ds_ud = f"{ds_ud:0.3f}".lstrip("0").rstrip("0")
            ds_udw = f"{ds_udw:0.3f}".lstrip("0").rstrip("0")
            mstr = "masked " if self.masked else ""
            istr = "imp " if self.impute_missing else ""
            row[1].set_title(f"{mstr}{istr} u{ds_ud} uw{ds_udw}", fontsize=7)

            if self.do_reg:
                sns.regplot(
                    x=dbad,
                    y=identity == u,
                    logistic=True,
                    color="k",
                    ax=row[2],
                    ci=None,
                )


class NearbyDivergencesMatrix(GMMMergePlot):
    kind = "amatrix"
    width = 2
    height = 2

    def __init__(
        self,
        cmap=plt.cm.rainbow,
        dist_vmax=1,
        merge_on_waveform_radius=True,
        n_neighbors=5,
        show_values=True,
        badness_kind=None,
    ):
        self.cmap = cmap
        self.dist_vmax = dist_vmax
        self.merge_on_waveform_radius = merge_on_waveform_radius
        self.n_neighbors = n_neighbors
        self.show_values = show_values
        self.badness_kind = badness_kind

    def draw(self, panel, gmm, unit_id):
        kind = self.badness_kind
        if kind is None:
            kind = gmm.merge_metric

        neighbors = self.get_neighbors(gmm, unit_id)
        nu = neighbors.size
        divergences = gmm.central_divergences(
            units_a=torch.from_numpy(neighbors),
            units_b=torch.from_numpy(neighbors),
            kind=kind,
            allow_chan_subset=self.merge_on_waveform_radius,
        )
        dists = divergences.numpy(force=True)

        axis = panel.subplots()
        im = axis.imshow(
            dists,
            vmin=0,
            vmax=self.dist_vmax,
            cmap=self.cmap,
            origin="lower",
            interpolation="none",
        )
        if self.show_values:
            for (j, i), label in np.ndenumerate(dists):
                axis.text(
                    i,
                    j,
                    f"{label:.2f}".lstrip("0"),
                    ha="center",
                    va="center",
                    clip_on=True,
                    fontsize=5,
                )
        plt.colorbar(im, ax=axis, shrink=0.3)
        axis.set_xticks(range(nu), neighbors)
        axis.set_yticks(range(nu), neighbors)
        for i, (tx, ty) in enumerate(
            zip(axis.xaxis.get_ticklabels(), axis.yaxis.get_ticklabels())
        ):
            tx.set_color(glasbey1024[neighbors[i]])
            ty.set_color(glasbey1024[neighbors[i]])
        chanstr = "reas" if self.merge_on_waveform_radius else "unit"
        title = f"{kind} ({chanstr} chans)"
        axis.set_title(title)


class ISICorner(GMMMergePlot):
    kind = "corner"
    width = 5
    height = 5

    def __init__(self, n_neighbors=5, bin_ms=0.1, max_ms=5, tick_step=1):
        self.n_neighbors = n_neighbors
        self.bin_ms = bin_ms
        self.max_ms = max_ms
        self.tick_step = tick_step

    def draw(self, panel, gmm, unit_id):
        # pick nearest units
        neighbors = self.get_neighbors(gmm, unit_id)
        colors = glasbey1024[neighbors % len(glasbey1024)]

        axes = panel.subplots(
            nrows=len(neighbors),
            ncols=len(neighbors),
            sharex=True,
            sharey=False,
            squeeze=False,
            gridspec_kw=dict(hspace=0),
        )
        bin_edges = np.arange(0, self.max_ms + self.bin_ms, self.bin_ms)
        for i, ua in enumerate(neighbors):
            in_ua = gmm.labels == ua
            times_s_a = gmm.data.times_seconds[in_ua].numpy(force=True)

            # diagonal axis is my ISI
            dt_ms_a = np.diff(times_s_a) * 1000
            counts, bin_edges = np.histogram(dt_ms_a, bin_edges)
            axes[i, i].stairs(counts, bin_edges, color=colors[i], fill=True)
            axes[i, i].text(
                0.5,
                0.9,
                f"{ua}: {times_s_a.size}sp",
                color="k",
                fontsize=5,
                ha="center",
                transform=axes[i, i].transAxes,
                backgroundcolor=(1, 1, 1, 0.75),
            )

            for j, ub in enumerate(neighbors):
                if j > i:
                    axes[i, j].axis("off")
                if j >= i:
                    continue

                in_ub = gmm.labels == ub
                times_s_b = gmm.data.times_seconds[in_ub].numpy(force=True)

                # visualize merged ISI
                times_s_ab = np.sort(np.concatenate((times_s_a, times_s_b)))
                dt_ms_ab = np.diff(times_s_ab) * 1000
                counts, bin_edges = np.histogram(dt_ms_ab, bin_edges)
                axes[i, j].stairs(counts, bin_edges, color="k", fill=True)
                # axes[i, j].set_yticks([])
                axes[i, j].set_ylim([0, max(1, counts.max())])
                # axes[i, j].text(
                #     0.1,
                #     0.9,
                #     f"{counts.max()}",
                #     color="k",
                #     fontsize=5,
                #     ha="center",
                #     transform=axes[i, j].transAxes,
                #     backgroundcolor=(1, 1, 1, 0.75),
                # )

                if i == len(neighbors) - 1:
                    axes[i, j].set_xlabel("isi (ms)")
                axes[i, j].set_xticks(np.arange(0, self.max_ms + self.tick_step, self.tick_step))

                
class CCGColumn(GMMMergePlot):
    kind = "neighbors"
    width = 3
    height = 10

    def __init__(self, n_neighbors=5, max_lag=50):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.max_lag = max_lag

    def draw(self, panel, gmm, unit_id):
        # pick nearest units
        neighbor_ids = self.get_neighbors(gmm, unit_id)
        colors = glasbey1024[neighbor_ids % len(glasbey1024)]

        axes = panel.subplots(
            nrows=self.n_neighbors + 1,
            ncols=2,
            sharex=True,
            sharey=False,
            squeeze=False,
        )

        my_st = gmm.data.times_samples[gmm.labels == unit_id]
        alags, acg = unit.correlogram(my_st, max_lag=self.max_lag)
        unit.bar(axes[0, 1], alags, acg, fill=True, fc=colors[0])
        axes[0, 0].axis("off")
        axes[0, 1].set_ylabel(f"my acg {neighbor_ids[0]}")

        j = 0
        for j, ub in enumerate(neighbor_ids[1:], start=1):
            their_st = gmm.data.times_samples[gmm.labels == ub]
            clags, ccg = unit.correlogram(my_st, their_st, max_lag=self.max_lag)
            merged_st = np.concatenate((my_st, their_st))
            merged_st.sort()
            alags, acg = unit.correlogram(merged_st, max_lag=self.max_lag)

            unit.bar(axes[j, 0], clags, ccg, fill=True, fc=colors[j])  # , ec="k", lw=1)
            unit.bar(axes[j, 1], alags, acg, fill=True, fc=colors[j])  # , ec="k", lw=1)
            axes[j, 0].set_ylabel(f"ccg {ub}")
            axes[j, 1].set_ylabel(f"macg {ub}")

        axes[j, 1].set_xlabel("lag (samples)")
        if j > 0:
            axes[j, 0].set_xlabel("lag (samples)")

        for k in range(j + 1, self.n_neighbors):
            axes[k, 0].axis("off")
            axes[k, 1].axis("off")


default_gmm_plots = (
    ISIHistogram(),
    ChansHeatmap(),
    MStep(),
    # HDBScanSplitPlot(spike_kind="residual_full"),
    # HDBScanSplitPlot(),
    # ZipperSplitPlot(),
    KMeansPPSPlitPlot(),
    # GridMeansSingleChanPlot(),
    InputWaveformsSingleChanPlot(),
    # InputWaveformsSingleChanOverTimePlot(channel="unit"),
    # InputWaveformsSingleChanOverTimePlot(channel="natural"),
    # ResidualsSingleChanPlot(),
    AmplitudesOverTimePlot(),
    BadnessesOverTimePlot(),
    # EmbedsOverTimePlot(),
    # DPCSplitPlot(spike_kind="residual_full"),
    # DPCSplitPlot(spike_kind="split"),
    # DPCSplitPlot(spike_kind="global"),
    # DPCSplitPlot(spike_kind="global", feature="spread_amp"),
    FeaturesVsBadnessesPlot(),
    # GridMeanDistancesPlot(),
    # GridMeansMultiChanPlot(),
    # InputWaveformsMultiChanPlot(),
)


gmm_merge_plots = (
    NearbyMeansSingleChan(),
    NearbyMeansMultiChan(),
    ViolatorTimesVAmps(),
    NearbyTimesVAmps(),
    NearbyDivergencesMatrix(merge_on_waveform_radius=True, badness_kind="diagz"),
    # NearbyDivergencesMatrix(merge_on_waveform_radius=False, badness_kind="diagz"),
    NearbyDivergencesMatrix(merge_on_waveform_radius=True, badness_kind="max1-r^2"),
    NearbyDivergencesMatrix(merge_on_waveform_radius=True, badness_kind="cos"),
    NearbyDivergencesMatrix(merge_on_waveform_radius=True, badness_kind="l2normeucsq"),
    # NearbyDivergencesMatrix(merge_on_waveform_radius=True, badness_kind="1-scaledr^2"),
    # NearbyDivergencesMatrix(merge_on_waveform_radius=False, badness_kind="diagz"),
    # NearbyDivergencesMatrix(merge_on_waveform_radius=False, badness_kind="1-scaledr^2"),
    ViolatorTimesVBadness(),
    NearbyTimesVBadness(),
    ISICorner(bin_ms=0.25),
    ISICorner(bin_ms=0.5, max_ms=8, tick_step=2),
    # NeighborBimodality(),
    # NeighborBimodality(badness_kind=None, masked=True, impute_missing=True, kind="truncnorm"),
    NeighborBimodality(badness_kind=None, masked=True, impute_missing=True),
    CCGColumn(),
    # NeighborBimodality(badness_kind="1-scaledr^2", masked=True),
)

gmm_selected_plots = (
    NearbyMeansMultiChan(),
    NearbyDivergencesMatrix(merge_on_waveform_radius=True),
    MStep(),
    ChansHeatmap(),
    KMeansPPSPlitPlot(),
    AmplitudesOverTimePlot(),
)


def make_unit_gmm_summary(
    gmm,
    unit_id,
    plots=default_gmm_plots,
    max_height=11,
    figsize=(11, 8.5),
    hspace=0.1,
    figure=None,
    **other_global_params,
):
    # notify plots of global params
    for p in plots:
        p.notify_global_params(
            time_range=gmm.t_bounds,
            **other_global_params,
        )

    figure = layout.flow_layout(
        plots,
        max_height=max_height,
        figsize=figsize,
        hspace=hspace,
        figure=figure,
        gmm=gmm,
        unit_id=unit_id,
    )

    return figure


def make_all_gmm_merge_summaries(
    gmm,
    save_folder,
    plots=gmm_merge_plots,
    max_height=11,
    figsize=(22, 10),
    dpi=250,
    **kwargs,
):
    return make_all_gmm_summaries(
        gmm, save_folder, plots=plots, max_height=max_height, figsize=figsize, dpi=dpi, **kwargs
    )


def make_all_gmm_summaries(
    gmm,
    save_folder,
    plots=default_gmm_plots,
    max_height=11,
    figsize=(18, 10),
    hspace=0.1,
    dpi=200,
    image_ext="png",
    n_jobs=0,
    show_progress=True,
    overwrite=False,
    unit_ids=None,
    use_threads=False,
    n_units=None,
    seed=0,
    **other_global_params,
):
    save_folder = Path(save_folder)
    if unit_ids is None:
        unit_ids = gmm.unit_ids().numpy(force=True)
    if n_units is not None and n_units < len(unit_ids):
        rg = np.random.default_rng(seed)
        unit_ids = rg.choice(unit_ids, size=n_units, replace=False)
    if not overwrite and all_summaries_done(unit_ids, save_folder, ext=image_ext):
        return

    save_folder.mkdir(exist_ok=True)

    global_params = dict(
        **other_global_params,
    )

    ispar = n_jobs > 0
    cls = CloudpicklePoolExecutor
    if use_threads:
        cls = ThreadPoolExecutor
    n_jobs, Executor, context = get_pool(n_jobs, cls=cls)
    from cloudpickle import dumps

    initargs = (
        gmm,
        plots,
        max_height,
        figsize,
        hspace,
        dpi,
        save_folder,
        image_ext,
        overwrite,
        global_params,
    )
    if ispar and not use_threads:
        initargs = (dumps(initargs),)
    with Executor(
        max_workers=n_jobs,
        mp_context=context,
        initializer=_summary_init,
        initargs=initargs,
    ) as pool:
        results = pool.map(_summary_job, unit_ids)
        if show_progress:
            results = tqdm(
                results,
                desc="GMM summaries",
                smoothing=0,
                total=len(unit_ids),
            )
        for res in results:
            pass


def all_summaries_done(unit_ids, save_folder, ext="png"):
    return save_folder.exists() and all(
        (save_folder / f"unit{unit_id:04d}.{ext}").exists() for unit_id in unit_ids
    )


class SummaryJobContext:
    def __init__(
        self,
        gmm,
        plots,
        max_height,
        figsize,
        hspace,
        dpi,
        save_folder,
        image_ext,
        overwrite,
        global_params,
    ):
        self.gmm = gmm
        self.plots = plots
        self.max_height = max_height
        self.figsize = figsize
        self.hspace = hspace
        self.dpi = dpi
        self.save_folder = save_folder
        self.image_ext = image_ext
        self.overwrite = overwrite
        self.global_params = global_params


_summary_job_context = None


def _summary_init(*args):
    global _summary_job_context
    if len(args) == 1:
        from cloudpickle import loads

        args = loads(args[0])
    _summary_job_context = SummaryJobContext(*args)


def _summary_job(unit_id):
    # handle resuming/overwriting
    try:
        ext = _summary_job_context.image_ext
        tmp_out = _summary_job_context.save_folder / f"tmp_unit{unit_id:04d}.{ext}"
        final_out = _summary_job_context.save_folder / f"unit{unit_id:04d}.{ext}"
        if tmp_out.exists():
            tmp_out.unlink()
        if not _summary_job_context.overwrite and final_out.exists():
            return
        if _summary_job_context.overwrite and final_out.exists():
            final_out.unlink()

        fig = plt.figure(
            figsize=_summary_job_context.figsize,
            layout="constrained",
            # dpi=_summary_job_context.dpi,
        )
        make_unit_gmm_summary(
            _summary_job_context.gmm,
            unit_id,
            hspace=_summary_job_context.hspace,
            plots=_summary_job_context.plots,
            max_height=_summary_job_context.max_height,
            figsize=_summary_job_context.figsize,
            figure=fig,
            **_summary_job_context.global_params,
        )

        # the save is done sort of atomically to help with the resuming and avoid
        # half-baked image files
        fig.savefig(tmp_out, dpi=_summary_job_context.dpi)
        tmp_out.rename(final_out)
        plt.close(fig)
    except Exception:
        import traceback

        print("// error in unit", unit_id)
        print(traceback.format_exc())
        raise
    finally:
        if tmp_out.exists():
            tmp_out.unlink()


def mad(x, axis=None):
    x = x - np.median(x, axis=axis, keepdims=True)
    np.abs(x, out=x)
    return np.median(x, axis=axis)


def plot_input_waveforms(
    gmm,
    which,
    waveform_kind="original",
    max_abs_amp=None,
    lw=1,
    color=None,
    colors=None,
    show_zero=False,
    subar=True,
    msbar=False,
    zlim="tight",
    ax=None,
    **more_geomplot_kwargs,
):
    data = gmm.spike_data(which, waveform_kind=waveform_kind)
    geom = gmm.data.registered_geom

    waveforms = data['waveforms']
    waveform_channels = data['waveform_channels']
    # times = data['times']

    n, r, c = waveforms.shape
    waveforms = waveforms.mT.reshape(n * c, r)
    waveforms = gmm.data.tpca._inverse_transform_in_probe(waveforms)
    waveforms = waveforms.view(n, c, -1).mT

    geomplot(
        waveforms.numpy(force=True),
        channels=waveform_channels.numpy(force=True),
        geom=geom.numpy(force=True),
        max_abs_amp=max_abs_amp,
        lw=lw,
        color=color,
        colors=colors,
        show_zero=show_zero,
        subar=subar,
        msbar=msbar,
        zlim=zlim,
        ax=ax,
        **more_geomplot_kwargs,
    )


def plot_raw_waveforms(
    gmm,
    original_sorting,
    rec,
    which,
    waveform_kind="original",
    max_abs_amp=None,
    lw=1,
    color=None,
    colors=None,
    show_zero=False,
    subar=True,
    msbar=False,
    zlim="tight",
    ax=None,
    **more_geomplot_kwargs,
):
    main_channels = original_sorting.channels[gmm.data.keepers[which]]

    waveforms = spikeio.read_waveforms_channel_index(
        rec,
        original_sorting.times_samples[gmm.data.keepers[which]],
        main_channels=main_channels,
        channel_index=gmm.data.original_channel_index,
    )

    geomplot(
        waveforms,
        max_channels=main_channels,
        channel_index=gmm.data.original_channel_index,
        geom=rec.get_channel_locations(),
        max_abs_amp=max_abs_amp,
        lw=lw,
        color=color,
        colors=colors,
        show_zero=show_zero,
        subar=subar,
        msbar=msbar,
        zlim=zlim,
        ax=ax,
        **more_geomplot_kwargs,
    )
