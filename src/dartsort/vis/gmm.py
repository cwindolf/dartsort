from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
from tqdm.auto import tqdm

from ..cluster import density
from .colors import glasbey1024
from . import analysis_plots, layout
from ..util.multiprocessing_util import CloudpicklePoolExecutor, get_pool
from .waveforms import geomplot
from ..util.waveform_util import grab_main_channels

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
    height = 2

    def __init__(self, spike_kind="residual", feature="pca"):
        self.spike_kind = spike_kind
        assert feature in ("pca", "spread_amp")
        self.feature = feature

    def draw(self, panel, gmm, unit_id):
        if self.feature == "pca":
            if self.spike_kind == "residual_full":
                (in_unit,) = (gmm.labels == unit_id).nonzero(as_tuple=True)
                features = torch.empty((in_unit.numel(), gmm.residual_pca_rank), device=gmm.device)
                for sl, data in gmm.batches(in_unit):
                    gmm[unit_id].residual_embed(**data,  out=features[sl])
                z = features[:, :gmm.dpc_split_kw.rank].numpy(force=True)
            elif self.spike_kind == "train":
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
            assert self.spike_kind in ("train", "global")
            in_unit, data = gmm.get_training_data(unit_id)
            waveforms = data["waveforms"]
            channel_norms = torch.sqrt(torch.nan_to_num(waveforms.square().sum(1)))
            amp = channel_norms.max(1).values
            logs = torch.nan_to_num(channel_norms.log())
            spread = (channel_norms * logs).sum(1)
            z = np.c_[amp.numpy(force=True), spread.numpy(force=True)]
            z /= mad(z, 0)
            print(f"{amp.min()=} {amp.max()=} {amp.mean()=} {amp.std()=}")
            print(f"{spread.min()=} {spread.max()=} {spread.mean()=} {spread.std()=}")

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
            ax.title("all spikes binned")
            ax.scatter(*z[:, :2].T, s=5, lw=0, color="k")
            ax.autoscale_view()
            return

        ru = np.unique(dens["labels"])
        panel, axes = analysis_plots.density_peaks_study(
            z,
            dens,
            s=10,
            idx=idx,
            inv=inv,
            fig=panel,
        )
        axes[-1].set_title(f"n={(ru>=0).sum()}", fontsize=8)
        axes[0].set_title(self.spike_kind)
        if self.feature == "spread_amp":
            axes[0].set_xlabel("spread")
            axes[0].set_ylabel("amp")



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
        amps = np.nan_to_num(gmm.data.static_amp_vecs[in_unit]).ptp(1)
        z = np.c_[times / mad(times), amps / mad(amps)]
        dens = density.density_peaks_clustering(
            z,
            # sigma_local=gmm.dpc_split_kw.sigma_local,
            sigma_local=0.5,
            sigma_regional=1.,
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
        plt.colorbar(im, ax=axis, shrink=0.3)
        axis.set_xticks(range(len(new_units)))
        axis.set_yticks(range(len(new_units)))
        for i, (tx, ty) in enumerate(
            zip(axis.xaxis.get_ticklabels(), axis.yaxis.get_ticklabels())
        ):
            tx.set_color(glasbey1024[i])
            ty.set_color(glasbey1024[i])
        axis.set_title(gmm.merge_metric)


class KMeansPPSPlitPlot(GMMPlot):
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
        n_clust=5,
        n_iter=20,
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
        self.n_clust = n_clust
        self.n_iter = n_iter
        if self.scaled:
            self.title = f"scaled {self.title}"

    def draw(self, panel, gmm, unit_id):
        in_unit, labels = gmm.kmeanspp(unit_id, n_clust=self.n_clust, n_iter=self.n_iter)

        times = gmm.data.times_seconds[in_unit].numpy(force=True)
        amps = np.nan_to_num(gmm.data.static_amp_vecs[in_unit]).ptp(1)
        labels = labels.numpy(force=True)
        ids = np.unique(labels)
        ids = ids[ids >= 0]

        if ids.size > 1:
            top, bottom = panel.subfigures(nrows=2)
            ax_top = top.subplots()
            axes = bottom.subplot_mosaic("de")
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

        new_units = []
        for label in ids:
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
            u.fit_center(**train_data, show_progress=False)
            new_units.append(u)
        
        ju = [(j, u) for j, u in enumerate(new_units) if u.n_chans_unit]

        # plot new unit maxchan wfs and old one in black
        ax = axes["d"]
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

            chans = torch.full((times.numel(),), unit.max_channel, device=times.device)
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

        # plot distance matrix
        kind = gmm.merge_metric
        min_overlap = gmm.min_overlap
        subset_channel_index = None
        if gmm.merge_on_waveform_radius:
            subset_channel_index = gmm.data.registered_reassign_channel_index
        nu = len(new_units)
        divergences = torch.full((nu, nu), torch.nan)
        for i, ua in ju:
            for j, ub in ju:
                if i == j:
                    divergences[i, j] = 0
                    continue
                divergences[i, j] = ua.divergence(
                    ub,
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
                axis.text(i, j, f"{label:.2f}".lstrip("0"), ha="center", va="center", clip_on=True, fontsize=5)
        plt.colorbar(im, ax=axis, shrink=0.3)
        axis.set_xticks(range(len(new_units)))
        axis.set_yticks(range(len(new_units)))
        for i, (tx, ty) in enumerate(
            zip(axis.xaxis.get_ticklabels(), axis.yaxis.get_ticklabels())
        ):
            tx.set_color(glasbey1024[i])
            ty.set_color(glasbey1024[i])
        axis.set_title(gmm.merge_metric)


class HDBScanSplitPlot(GMMPlot):
    kind = "split"
    width = 2
    height = 2

    def __init__(self, spike_kind="train"):
        self.spike_kind = spike_kind

    def draw(self, panel, gmm, unit_id):
        if self.spike_kind == "residual_full":
            (in_unit,) = (gmm.labels == unit_id).nonzero(as_tuple=True)
            features = torch.empty((in_unit.numel(), gmm.residual_pca_rank), device=gmm.device)
            for sl, data in gmm.batches(in_unit):
                gmm[unit_id].residual_embed(**data,  out=features[sl])
            z = features[:, :gmm.dpc_split_kw.rank].numpy(force=True)
        elif self.spike_kind == "train":
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
    width = 5
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
        ax.scatter(*xy[unique_ixs].T, c=counts, lw=0, cmap=self.cmap)
        ax.scatter(*xy[np.atleast_1d(gmm[unit_id].max_channel.numpy(force=True))].T, color="g", lw=0)


class AmplitudesOverTimePlot(GMMPlot):
    kind = "embeds"
    width = 5
    height = 3

    def draw(self, panel, gmm, unit_id):
        in_unit, utd = gmm.get_training_data(unit_id, waveform_kind="reassign")
        amps = utd["static_amp_vecs"].numpy(force=True)
        amps = np.nanmax(amps, axis=1)
        amps2 = utd["waveforms"]
        n, r, c = amps2.shape
        amps2 = gmm.data.tpca._inverse_transform_in_probe(amps2.permute(0, 2, 1).reshape(n * c, r))
        amps2 = amps2.reshape(n, -1, c).permute(0, 2, 1)
        amps2 = np.nan_to_num(amps2.numpy(force=True)).ptp(axis=(1, 2))
        recons = gmm[unit_id].get_means(utd["times"])
        recons = gmm[unit_id].to_waveform_channels(recons, waveform_channels=utd["waveform_channels"])
        n, r, c = recons.shape
        recons = gmm.data.tpca._inverse_transform_in_probe(recons.permute(0, 2, 1).reshape(n * c, r))
        recons = recons.reshape(n, -1, c).permute(0, 2, 1)

        recon_amps = np.nan_to_num(recons.numpy(force=True)).ptp(axis=(1, 2))

        ax = panel.subplots()
        ax.scatter(utd["times"].numpy(force=True), amps, s=3, c="b", lw=0, label='observed (final)')
        ax.scatter(utd["times"].numpy(force=True), amps2, s=3, c="k", lw=0, label='observed (feat)')
        ax.scatter(utd["times"].numpy(force=True), recon_amps, s=3, c="r", lw=0, label='model')
        # ax.legend(loc="upper left", ncols=3)
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
        plt.stairs(counts, bin_edges, color="k", fill=True)
        axis.set_xlabel("isi (ms)")
        axis.set_ylabel(f"count (out of {dt_ms.size} total isis)")


class BadnessesOverTimePlot(GMMPlot):
    kind = "embeds"
    width = 5
    height = 3

    def __init__(self, colors="rgb", kinds=("1-r^2", "1-scaledr^2")):
        self.colors = colors
        self.kinds = kinds

    def draw(self, panel, gmm, unit_id):
        in_unit, utd = gmm.get_training_data(unit_id, waveform_kind="reassign")

        spike_ix, overlaps, badnesses = gmm[unit_id].spike_badnesses(
            utd["times"],
            utd["waveforms"],
            utd["waveform_channels"],
            kinds=self.kinds,
        )
        overlaps = overlaps.numpy(force=True)
        times = utd["times"][spike_ix].numpy(force=True)

        ax = panel.subplots()
        for j, (kind, b) in enumerate(badnesses.items()):
            ax.scatter(
                times,
                b.numpy(force=True),
                alpha=overlaps,
                s=3,
                c=self.colors[j],
                lw=0,
                label=kind,
            )
        ax.legend(loc="upper left")
        ax.set_ylabel("badness")


class FeaturesVsBadnessesPlot(GMMPlot):
    kind = "triplet"
    width = 5
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
        overlaps = overlaps.numpy(force=True)
        badnesses = {k: v.numpy(force=True) for k, v in badnesses.items()}

        # amplitudes
        amps = utd["static_amp_vecs"].numpy(force=True)
        amps = np.nanmax(amps, axis=1)

        axes = panel.subplots(ncols=z.shape[1] + 1, sharey=True)
        for ax, feat, featname in zip(
            axes,
            (amps, *z.T),
            ("amps", *[f"emb{x}" for x in range(z.shape[1])]),
        ):
            for j, kind in enumerate(self.kinds):
                ax.scatter(feat, badnesses[kind], color=self.colors[j], s=3, lw=0)
            ax.set_xlabel(featname)
        axes[0].set_ylabel('badnesses')


class GridMeansMultiChanPlot(GMMPlot):
    kind = "waveform"
    width = 5
    height = 5

    def __init__(self, cmap=plt.cm.rainbow, time_range=None, fitted_only=True):
        self.cmap = cmap
        self.time_range = time_range
        self.fitted_only = fitted_only

    def draw(self, panel, gmm, unit_id):
        # amps = np.nanmax(amps, axis=1)
        if gmm[unit_id].do_interp:
            times = gmm[unit_id].interp.grid.squeeze()
            if self.fitted_only:
                times = times[gmm[unit_id].interp.grid_fitted]
        else:
            times = torch.tensor([sum(gmm.t_bounds) / 2]).to(gmm.device)
        times = torch.atleast_1d(times)
        time_range = self.time_range
        if time_range is None:
            time_range = times.min(), times.max()
        chans = torch.full(times.shape, gmm[unit_id].max_channel, device=times.device)
        waveform_channels = gmm.data.registered_reassign_channel_index[chans]
        if waveform_channels.ndim == 1:
            waveform_channels = waveform_channels[None]
        recons = gmm[unit_id].get_means(times)
        recons = gmm[unit_id].to_waveform_channels(recons, waveform_channels=waveform_channels)
        recons = gmm.data.tpca.inverse_transform(recons, chans, channel_index=gmm.data.registered_reassign_channel_index)
        recons = recons.numpy(force=True)
        chans = chans.numpy(force=True)
        if gmm[unit_id].do_interp:
            c = times.numpy(force=True)
            c = (c - time_range[0]) / (time_range[1] - time_range[0])
            colors = self.cmap(c)
        else:
            colors = "r"
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
        if gmm[unit_id].do_interp:
            times = gmm[unit_id].interp.grid.squeeze()
            if self.fitted_only:
                times = times[gmm[unit_id].interp.grid_fitted]
        else:
            times = torch.tensor([sum(gmm.t_bounds) / 2]).to(gmm.device)
        times = torch.atleast_1d(times)
        time_range = self.time_range
        if time_range is None:
            time_range = times.min(), times.max()
        chans = torch.full((times.numel(),), gmm[unit_id].max_channel, device=times.device)
        recons = gmm[unit_id].get_means(times).to(gmm.device)
        recons = gmm[unit_id].to_waveform_channels(recons, waveform_channels=chans[:, None])
        recons = recons[..., 0]
        recons = gmm.data.tpca._inverse_transform_in_probe(recons)
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


class InputWaveformsMultiChanPlot(GMMPlot):
    kind = "waveform"
    width = 5
    height = 5

    def __init__(self, cmap=plt.cm.rainbow, max_plot=250, rg=0, time_range=None):
        self.cmap = cmap
        self.max_plot = max_plot
        self.rg = np.random.default_rng(0)
        self.time_range = time_range

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
        waveforms = gmm.data.tpca._inverse_transform_in_probe(waveforms.permute(0, 2, 1).reshape(n * c, r))
        waveforms = waveforms.reshape(n, c, -1).permute(0, 2, 1)
        waveforms = waveforms.numpy(force=True)
        c = times.numpy(force=True)
        c = (c - time_range[0]) / (time_range[1] - time_range[0])
        colors = self.cmap(c)
        maa = np.nanmax(np.abs(waveforms))

        ax = panel.subplots()
        geomplot(
            waveforms,
            channels=waveform_channels.numpy(force=True),
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
        chans = gmm.data.channels[in_unit[wh]]
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
            (np.broadcast_to(np.arange(waveforms.shape[1])[None], waveforms.shape), waveforms),
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

    def __init__(self, cmap=plt.cm.rainbow, max_plot=100, rg=0, time_range=None, dt=300, max_bins=15, channel="unit"):
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
            times_valid = gmm.data.times_seconds == gmm.data.times_seconds.clip(bin_left, bin_right)
            in_unit = torch.logical_and(
                gmm.labels == unit_id, 
                times_valid.to(gmm.labels),
            )
            (in_unit,) = torch.nonzero(in_unit, as_tuple=True)

            in_unit, utd = gmm.get_training_data(unit_id, n=self.max_plot, in_unit=in_unit, waveform_kind="reassign")
            times = torch.atleast_1d(utd["times"].squeeze())
            waveforms = utd["waveforms"]
            waveform_channels = utd["waveform_channels"]

            if self.channel == "unit":
                main_channel = gmm[unit_id].max_channel.to(waveform_channels)
                nixs, cixs = torch.nonzero(waveform_channels == main_channel, as_tuple=True)
            elif self.channel == "natural":
                chans = gmm.data.static_main_channels[in_unit].to(waveform_channels)
                nixs, cixs = torch.nonzero(waveform_channels == chans[:, None], as_tuple=True)
            else:
                assert False
            waveforms = waveforms[nixs]
            times = times[nixs]
            waveforms = torch.take_along_dim(waveforms, cixs[:, None, None], dim=2)[:, :, 0]

            waveforms = gmm.data.tpca._inverse_transform_in_probe(waveforms)
            c = times.numpy(force=True)
            c = (c - time_range[0]) / (time_range[1] - time_range[0])
            colors = self.cmap(c)
            waveforms = waveforms.numpy(force=True)

            ax.axhline(0, c="k", lw=0.8)
            lines = np.stack(
                (np.broadcast_to(np.arange(waveforms.shape[1])[None], waveforms.shape), waveforms),
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
        recons = gmm[unit_id].to_waveform_channels(recons, waveform_channels=waveform_channels)
        recons = torch.nan_to_num(recons * mask)

        scalings = torch.ones_like(recons)
        if self.scaled:
            dots = recons.mul(waveforms).sum(dim=(1, 2))
            recons_sumsq = recons.square().sum(dim=(1, 2))
            scalings = (dots + self.inv_lambda).div_(recons_sumsq + self.inv_lambda)
            scalings = scalings.clip_(self.scale_clip_low, self.scale_clip_high)[:, None, None]
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
            (np.broadcast_to(np.arange(waveforms.shape[1])[None], waveforms.shape), waveforms),
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

            dists = means[:, None].sub(scalings[:, :, None] * means[None]).square().sum(2).div(l2s)
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
        plt.colorbar(im, ax=axis, shrink=0.3)
        axis.set_xticks(range(len(times)), [f"{t:0.1f}" for t in times])
        axis.set_yticks(range(len(times)), [f"{t:0.1f}" for t in times])
        axis.set_title(self.title)


default_gmm_plots = (
    ISIHistogram(),
    ChansHeatmap(),
    # HDBScanSplitPlot(spike_kind="residual_full"),
    # HDBScanSplitPlot(),
    ZipperSplitPlot(),
    KMeansPPSPlitPlot(),
    GridMeansSingleChanPlot(),
    InputWaveformsSingleChanPlot(),
    InputWaveformsSingleChanOverTimePlot(channel="unit"),
    InputWaveformsSingleChanOverTimePlot(channel="natural"),
    ResidualsSingleChanPlot(),
    AmplitudesOverTimePlot(),
    BadnessesOverTimePlot(),
    EmbedsOverTimePlot(),
    DPCSplitPlot(spike_kind="residual_full"),
    DPCSplitPlot(spike_kind="train"),
    DPCSplitPlot(spike_kind="global"),
    DPCSplitPlot(spike_kind="global", feature="spread_amp"),
    FeaturesVsBadnessesPlot(),
    GridMeanDistancesPlot(),
    GridMeansMultiChanPlot(),
    InputWaveformsMultiChanPlot(),
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
    **other_global_params,
):
    save_folder = Path(save_folder)
    if unit_ids is None:
        unit_ids = gmm.unit_ids().numpy(force=True)
    if not overwrite and all_summaries_done(
        unit_ids, save_folder, ext=image_ext
    ):
        return

    save_folder.mkdir(exist_ok=True)

    global_params = dict(
        **other_global_params,
    )

    n_jobs, Executor, context = get_pool(n_jobs, cls=CloudpicklePoolExecutor)
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
    with Executor(
        max_workers=n_jobs,
        mp_context=context,
        initializer=_summary_init,
        initargs=(dumps(initargs),),
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


def _summary_init(args):
    global _summary_job_context
    from cloudpickle import loads

    args = loads(args)
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
    finally:
        if tmp_out.exists():
            tmp_out.unlink()


def mad(x, axis=None):
    x = x - np.median(x, axis=axis, keepdims=True)
    np.abs(x, out=x)
    return np.median(x, axis=axis)
