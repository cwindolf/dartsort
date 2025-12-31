from pathlib import Path
import math

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from tqdm.auto import tqdm

from ..cluster.gmm.mixture import (
    Scores,
    StreamingSpikeData,
    TruncatedMixtureModel,
    TruncatedSpikeData,
    instantiate_and_bootstrap_tmm,
    labels_from_scores,
)
from ..transform import TemporalPCA
from ..util import spiketorch
from ..util.data_util import DARTsortSorting, resolve_path, get_tpca
from ..util.internal_config import (
    ComputationConfig,
    RefinementConfig,
    default_refinement_cfg,
)
from ..util.job_util import ensure_computation_config
from ..util.multiprocessing_util import CloudpicklePoolExecutor, get_pool
from ..util.py_util import databag
from .colors import glasbey1024
from .layout import BasePlot, flow_layout
from .waveforms import geomplot
from .analysis_plots import distance_matrix_dendro


@databag
class MixtureVisData:
    """Collection passed around to the plots in this file, with helper methods for vis."""

    tmm: TruncatedMixtureModel
    train_data: TruncatedSpikeData
    val_data: TruncatedSpikeData | None
    full_data: StreamingSpikeData
    sorting: DARTsortSorting
    train_scores: Scores
    full_scores: Scores
    eval_scores: Scores
    train_times: np.ndarray
    train_labels: np.ndarray
    full_labels: np.ndarray
    tpca: TemporalPCA
    inf_diag_unit_distance_matrix: torch.Tensor
    prgeom: np.ndarray

    @property
    def times_seconds(self):
        return self.sorting.times_seconds  # type: ignore

    def inu_and_times_full(self, unit_id: int):
        inu_full = np.flatnonzero(self.full_labels == unit_id)
        times = self.times_seconds[inu_full]
        return inu_full, times

    def train_inds_and_chans(
        self, unit_id: int, count=None
    ) -> tuple[np.ndarray, np.ndarray]:
        inu_train = np.flatnonzero(self.train_labels == unit_id)
        if count and inu_train.size > count:
            rg = np.random.default_rng(0)
            inu_train = rg.choice(inu_train, size=count, replace=False)
            inu_train.sort()

        neighb_ids = self.train_data.neighborhood_ids[inu_train].cpu()
        chans = self.tmm.neighb_cov.obs_ix.cpu()[neighb_ids]
        return inu_train, chans.numpy(force=True)

    def chans_in_radius(self, unit_id: int, radius: float):
        mean = self.tmm.b.means[unit_id].view(self.tmm.neighb_cov.feat_rank, -1)
        my_chan = mean.square().sum(0).argmax()
        my_xy = self.tmm.neighb_cov.prgeom[my_chan]
        dxy = self.tmm.neighb_cov.prgeom - my_xy
        inf_dist = dxy.abs_().amax(dim=1)
        (close,) = (inf_dist < radius).cpu().nonzero(as_tuple=True)
        return close

    def friends(self, unit_id: int, count: int, me_last=True):
        dists, neighbors = self.inf_diag_unit_distance_matrix[unit_id].sort()
        count = min(count, dists.numel() - 1)
        assert count >= 0
        if count == 0:
            return np.zeros(1), np.array([unit_id])
        dists = dists[:count]
        neighbors = neighbors[:count]
        assert not (neighbors == unit_id).any()
        assert dists.isfinite().all()
        neighbors = np.concatenate([[unit_id], neighbors.numpy(force=True)])
        dists = np.concatenate([[0.0], dists.numpy(force=True)])
        if me_last:
            return dists[::-1], neighbors[::-1]
        else:
            return dists, neighbors

    def reconstruct_flat(self, features: torch.Tensor) -> np.ndarray:
        frank = self.tmm.neighb_cov.feat_rank
        single = features.ndim == 1
        if single:
            features = features[None]
        assert features.ndim == 2
        n = features.shape[0]
        features = features.view(n, frank, -1)
        recon = self.tpca.force_reconstruct(features)
        if single:
            recon = recon[0]
        return recon.numpy(force=True)

    def random_train_waveforms(
        self, unit_id: int, count=128
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        inu_train, chans = self.train_inds_and_chans(unit_id, count=count)
        features = self.train_data.x[inu_train]
        waveforms = self.reconstruct_flat(features)
        features = features.numpy(force=True)
        return inu_train, chans, features, waveforms


class MixtureComponentPlot(BasePlot):
    width = 1
    height = 1
    kind = "mixture"

    def draw(self, panel, mix_data: MixtureVisData, unit_id: int):
        raise NotImplementedError


class TextInfo(MixtureComponentPlot):
    kind = "small"
    height = 0.75

    def draw(self, panel, mix_data: MixtureVisData, unit_id: int):
        axis = panel.subplots()
        axis.axis("off")
        msg = f"unit {unit_id}\n"
        axis.text(0, 0, msg, fontsize=5.5)


class ISIHistogram(MixtureComponentPlot):
    kind = "small"
    height = 0.75

    def __init__(self, bin_ms=0.1, max_ms=5):
        self.bin_ms = bin_ms
        self.max_ms = max_ms

    def draw(self, panel, mix_data: MixtureVisData, unit_id: int):
        axis = panel.subplots()
        inu_full, times_s = mix_data.inu_and_times_full(unit_id)
        dt_ms = np.diff(times_s) * 1000
        bin_edges = np.arange(0, self.max_ms + self.bin_ms, self.bin_ms)
        counts, _ = np.histogram(dt_ms, bin_edges)
        axis.stairs(counts, bin_edges, color=glasbey1024[unit_id], fill=True)
        axis.set_xlabel("isi (ms)")
        axis.set_ylabel(f"count ({dt_ms.size + 1} tot. sp.)")


class ChansHeatmap(MixtureComponentPlot):
    kind = "tall"
    height = 1.5

    def __init__(self, cmap="magma", snr_cmap="viridis"):
        self.cmap = plt.get_cmap(cmap)
        self.snr_cmap = plt.get_cmap(snr_cmap)

    def draw(self, panel, mix_data: MixtureVisData, unit_id: int):
        inu_train, chans = mix_data.train_inds_and_chans(unit_id)
        chans = chans[chans < mix_data.tmm.neighb_cov.n_channels]

        mean = mix_data.tmm.b.mean[unit_id].view(-1, mix_data.tmm.neighb_cov.n_channels)
        ptp = spiketorch.ptp(mean, dim=0).cpu()
        (support,) = ptp.nonzero(as_tuple=True)
        ptp = ptp[support]

        uchans, counts = np.unique(chans, return_counts=True)
        ax = panel.subplots()
        xy = mix_data.tmm.neighb_cov.prgeom.numpy(force=True)
        ax.scatter(*xy[support].T, c=ptp, lw=2, s=5)
        s = ax.scatter(*xy[uchans].T, c=counts, lw=0, cmap=self.cmap, s=5)
        plt.colorbar(s, ax=ax, shrink=0.3, label="chan count")
        ax.scatter(*xy[support[ptp.argmax()]].T, color="gold", lw=1, fc="none", s=5)
        ax.set_title(f"chans:{chans.min().item()}-{chans.max().item()}")


class LikelihoodsView(MixtureComponentPlot):
    kind = "block"
    width = 2
    height = 1

    def __init__(self, viol_ms=1.0):
        self.viol_ms = viol_ms

    def compute(self, mix_data: MixtureVisData, unit_id: int):
        inu, t = mix_data.inu_and_times_full(unit_id)
        my_ll = mix_data.full_scores.log_liks[:, 0][inu]
        noise_ll = mix_data.full_scores.log_liks[:, -1][inu]
        dt_ms = np.diff(t) * 1000
        viol = dt_ms <= self.viol_ms
        viol = np.logical_or(
            np.pad(viol, (1, 0), constant_values=False),
            np.pad(viol, (0, 1), constant_values=False),
        )
        viol = np.flatnonzero(viol)
        my_min = np.min(my_ll)
        my_max = np.max(my_ll)
        noise_min = np.min(noise_ll)
        noise_max = np.max(noise_ll)
        mn = min(my_min, noise_min)
        mx = max(my_max, noise_max)
        assert math.isfinite(mn)
        return my_ll, noise_ll, t, viol, mn, mx

    def draw(self, panel, mix_data: MixtureVisData, unit_id: int):
        my_ll, noise_ll, t, viol, mn, mx = self.compute(mix_data, unit_id)

        ax_time, ax_noise, ax_hist = panel.subplots(
            ncols=3, width_ratios=[3, 2, 1], sharey=True
        )
        ax_time.set_ylabel("log likelihood")
        ax_time.set_xlabel("time (s)")
        ax_noise.set_xlabel("noise loglik")
        ax_hist.set_xlabel("count")
        c = glasbey1024[unit_id]

        ax_time.scatter(t, noise_ll, s=3, lw=0, color="darkgray")
        ax_time.scatter(t, my_ll, s=3, lw=0, color=c)
        ax_time.scatter(t[viol], my_ll[viol], s=3, lw=0, color="k")

        ax_noise.scatter(noise_ll, my_ll, s=3, lw=0, color=c, zorder=1)
        ax_noise.plot([mn, mx], [mn, mx], color="k", lw=0.8, zorder=11)

        histk = dict(histtype="step", orientation="horizontal")
        _, bins, _ = ax_hist.hist(my_ll, color=c, label="unit", bins=64, **histk)
        ax_hist.hist(noise_ll, color="darkgray", label="noise", bins=bins, **histk)
        ax_hist.legend(
            loc="lower right",
            borderpad=0.1,
            frameon=False,
            borderaxespad=0.1,
            handletextpad=0.3,
            handlelength=1.0,
        )


class NeighborMeans(MixtureComponentPlot):
    kind = "block"
    width = 2
    height = 2

    def __init__(self, count=5, vis_radius=50.0):
        self.count = count
        self.vis_radius = vis_radius

    def compute(self, mix_data: MixtureVisData, unit_id: int):
        dists, neighbors = mix_data.friends(unit_id, count=self.count)
        k = len(neighbors)
        chans = mix_data.chans_in_radius(unit_id, self.vis_radius)
        means = mix_data.tmm.b.means[neighbors].view(
            k, -1, mix_data.tmm.neighb_cov.n_channels
        )
        means = means[:, :, chans].view(k, -1)
        means = mix_data.reconstruct_flat(means)
        return chans, neighbors, means

    def draw(self, panel, mix_data: MixtureVisData, unit_id: int):
        chans, neighbors, means = self.compute(mix_data, unit_id)
        colors = [glasbey1024[j] for j in neighbors]
        ax = panel.subplots()
        chans = chans[None].broadcast_to(len(means), *chans.shape)
        geomplot(
            waveforms=means,
            channels=chans.numpy(force=True),
            geom=mix_data.prgeom,
            show_zero=False,
            colors=colors,
            zlim=None,
            ax=ax,
            subar=True,
            annotate_z=True,
        )
        panel.legend(
            handles=[Line2D([0, 1], [0, 0], color=c) for c in colors],
            labels=list(neighbors),
            loc="outside upper center",
            frameon=False,
            ncols=min(5, len(neighbors)),
            fontsize="small",
            borderpad=0,
            labelspacing=0.25,
            handlelength=1.0,
            handletextpad=0.4,
            borderaxespad=0.0,
            columnspacing=1.0,
        )
        ax.axis("off")


class NeighborDistances(MixtureComponentPlot):
    kind = "block"
    width = 2
    height = 2

    def __init__(self, count=5, cmap="managua"):
        self.count = count
        self.cmap = plt.get_cmap(cmap)

    def compute(self, mix_data: MixtureVisData, unit_id: int):
        dists, neighbors = mix_data.friends(unit_id, count=self.count)
        d = mix_data.inf_diag_unit_distance_matrix[neighbors][:, neighbors]
        d = d.numpy(force=True)
        np.fill_diagonal(d, 0.0)
        return d, neighbors

    def draw(self, panel, mix_data: MixtureVisData, unit_id: int):
        d, neighbors = self.compute(mix_data, unit_id)
        distance_matrix_dendro(
            panel,
            d,
            unit_ids=neighbors,
            dendrogram_linkage=None,
            show_unit_labels=True,
            vmax=mix_data.tmm.max_distance,
            image_cmap=self.cmap,
            show_values=True,
        )


class MergeView(MixtureComponentPlot):
    kind = "block"
    width = 2
    height = 2

    def __init__(self):
        pass

    def compute(self, mix_data: MixtureVisData, unit_id: int):
        # get pair mask
        gsize = mix_data.tmm.max_group_size
        dists, neighbors = mix_data.friends(unit_id, count=gsize)
        d = mix_data.inf_diag_unit_distance_matrix[neighbors][:, neighbors]
        d.diagonal().fill_(0.0)
        pair_mask = d < mix_data.tmm.max_distance

        # determine connected components and re-order (stably)
        n_comps, labels = connected_components(pair_mask.numpy(force=True))
        if n_comps > 1:
            reorder = [np.flatnonzero(labels == l) for l in range(n_comps)]
            reorder = np.concatenate(reorder)
            pair_mask = pair_mask[reorder][:, reorder]
            neighbors = neighbors[reorder]

        # run it
        group_res = mix_data.tmm.try_merge_group(
            group=torch.asarray(neighbors),
            train_data=mix_data.train_data,
            eval_data=mix_data.val_data,
            scores=mix_data.eval_scores,
        )

        return neighbors, pair_mask, group_res

    def draw(self, panel, mix_data: MixtureVisData, unit_id: int):
        neighbors, pair_mask, group_res = self.compute(mix_data, unit_id)
        panel_mask, panel_info = panel.subfigures(ncols=2, width_ratios=[2, 1])
        distance_matrix_dendro(
            panel_mask,
            pair_mask,
            unit_ids=neighbors,
            dendrogram_linkage=None,
            show_unit_labels=True,
            image_cmap="binary",
            show_values=True,
        )
        ax_info = panel_info.subplots()
        ax_info.axis("off")
        if group_res is None:
            msg = "bail"
        else:
            propstr = ",".join(f"{p:0.2f}" for p in group_res.sub_proportions.cpu())
            msg = (
                f"part: {group_res.grouping.group_ids.tolist()}\n"
                f"imp: {group_res.improvement:.3f}\n"
                f"props: {propstr}"
            )
        ax_info.text(0, 0, msg, fontsize=5.5)


class MeanView(MixtureComponentPlot):
    kind = "bigblock"
    width = 3
    height = 3

    def __init__(self, n_waveforms_show=128, alpha=0.1, show_zero=True):
        self.n_waveforms_show = n_waveforms_show
        self.alpha = alpha
        self.show_zero = show_zero

    def compute(self, mix_data: MixtureVisData, unit_id: int):
        mean = mix_data.tmm.b.mean[unit_id]
        mean_recon = mix_data.reconstruct_flat(mean)
        mean_recon = mean_recon.reshape(1, -1, mix_data.tmm.neighb_cov.n_channels)
        inu_train, wchans, features, waveforms = mix_data.random_train_waveforms(
            unit_id=unit_id, count=self.n_waveforms_show
        )
        maa = np.nanmax(np.abs(waveforms)).item()
        return waveforms, wchans, mean_recon, maa

    def draw(self, panel, mix_data: MixtureVisData, unit_id: int):
        waveforms, wchans, mean_recon, maa = self.compute(mix_data, unit_id)

        ax = panel.subplots()
        ax.axis("off")
        lines, pchans = geomplot(  # type: ignore
            waveforms=waveforms,
            channels=wchans,
            color="k",
            alpha=self.alpha,
            ax=ax,
            max_abs_amp=maa,
            geom=mix_data.tmm.neighb_cov.prgeom.numpy(force=True),
            show_zero=self.show_zero,
            return_chans=True,
        )
        pchans = np.array(list(pchans))
        geomplot(
            waveforms=mean_recon[:, :, pchans],
            channels=pchans[None],
            max_abs_amp=maa,
            geom=mix_data.tmm.neighb_cov.prgeom.numpy(force=True),
            color=glasbey1024[unit_id],
            ax=ax,
            show_zero=False,
            annotate_z=True,
        )


class CovarianceView(MixtureComponentPlot):
    kind = "bigblock"
    width = 3
    height = 4

    def __init__(self, n_waveforms_show=128, neigs=20):
        self.n_waveforms_show = n_waveforms_show
        self.colors = dict(emp="k", noise="r", signal="g", model="b")
        self.neigs = neigs

    def compute(self, mix_data: MixtureVisData, unit_id: int):
        # load data
        inu_train, wchans, features, waveforms = mix_data.random_train_waveforms(
            unit_id=unit_id, count=self.n_waveforms_show
        )

        # pick channels and put data there
        chan_set = torch.asarray(wchans).unique().to(device=mix_data.tmm.b.means.device)
        chan_set = chan_set[chan_set < mix_data.tmm.neighb_cov.n_channels]
        full_features = mix_data.tmm.erp.interp_to_chans(
            waveforms=torch.asarray(features).to(mix_data.tmm.b.means),
            neighborhood_ids=mix_data.train_data.neighborhood_ids[inu_train],
            target_channels=chan_set,
        )

        # empirical covariance
        cov_emp = torch.cov(full_features.T)

        # noise covariance
        cov_noise = mix_data.tmm.noise.marginal_covariance(channels=chan_set).to_dense()
        assert cov_noise.shape == cov_emp.shape

        # signal covariance
        if mix_data.tmm.signal_rank:
            basis = mix_data.tmm.b.bases[unit_id]
            basis = basis.view(
                mix_data.tmm.signal_rank, -1, mix_data.tmm.neighb_cov.n_channels
            )
            basis = basis[:, :, chan_set].view(mix_data.tmm.signal_rank, -1)
            cov_signal = basis.T @ basis
            assert cov_signal.shape == cov_noise.shape
        else:
            cov_signal = torch.zeros_like(cov_noise)

        # model covariance
        cov_model = cov_noise + cov_signal

        # np-ify
        cov_emp = cov_emp.numpy(force=True)
        cov_noise = cov_noise.numpy(force=True)
        cov_signal = cov_signal.numpy(force=True)
        cov_model = cov_model.numpy(force=True)

        # spectra: empirical, noise, sginal, model
        ev_emp = np.linalg.eigvalsh(cov_emp)[::-1]
        ev_noise = np.linalg.eigvalsh(cov_noise)[::-1]
        ev_signal = np.linalg.eigvalsh(cov_signal)[::-1]
        ev_model = np.linalg.eigvalsh(cov_model)[::-1]

        return {
            "emp": (cov_emp, ev_emp),
            "noise": (cov_noise, ev_noise),
            "signal": (cov_signal, ev_signal),
            "model": (cov_model, ev_model),
        }

    def draw(self, panel, mix_data: MixtureVisData, unit_id: int):
        stats = self.compute(mix_data, unit_id)
        cov_emp = stats["emp"][0]
        vm = np.percentile(np.abs(cov_emp), 98)

        top, bottom = panel.subfigures(nrows=2, height_ratios=[3, 1])
        axes_top = top.subplots(ncols=2, nrows=len(stats))
        ax_bottom = bottom.subplots()

        cov_kw = dict(
            aspect=1.0, interpolation="none", vmin=-vm, vmax=vm, cmap="seismic"
        )
        res_kw = cov_kw | dict(vmin=-0.1 * vm, vmax=0.1 * vm)

        for row_top, (name, (cov, ev)) in zip(axes_top, stats.items()):
            c = self.colors[name]
            row_top[0].set_ylabel(name, color=c)
            row_top[0].imshow(cov, **cov_kw)
            row_top[1].imshow(cov_emp - cov, **res_kw)
            ax_bottom.plot(ev[: self.neigs], color=c)

        axes_top[0, 0].set_title("cov", fontsize="small")
        axes_top[0, 1].set_title("emp - cov", fontsize="small")
        for ax in axes_top.flat:
            ax.axis("off")
        ax_bottom.grid(which="both")


class SplitView(MixtureComponentPlot):
    kind = "tall"
    width = 2.5
    height = 7

    def __init__(
        self,
        colors=["r", "g", "b"],
        bail_color="k",
        vis_radius=50.0,
        dist_cmap="managua",
    ):
        self.colors = np.array(colors)
        self.bail_color = bail_color
        self.vis_radius = vis_radius
        self.dist_cmap = plt.get_cmap(dist_cmap)

    def compute(self, mix_data: MixtureVisData, unit_id: int):
        split_res, debug_info = mix_data.tmm.split_unit(
            unit_id=unit_id,
            train_data=mix_data.train_data,
            eval_data=mix_data.val_data,
            scores=mix_data.eval_scores,
            debug=True,
        )
        assert debug_info is not None

        # compute text info
        if debug_info.bailed:
            txt = f"bail: {debug_info.bail_reason}"
        else:
            assert debug_info.merge_res is not None
            propstr = ",".join(
                f"{p:0.2f}" for p in debug_info.merge_res.sub_proportions.cpu()
            )
            txt = (
                f"part: {debug_info.merge_res.grouping.group_ids.tolist()}\n"
                f"imp: {debug_info.merge_res.improvement:.3f}\n"
                f"props: {propstr}"
            )

        if debug_info.split_data is not None:
            n_spikes = debug_info.split_data.x.shape[0]
        else:
            n_spikes = 0

        # compute colors by kmeans label
        if debug_info.kmeans_responsibilities is not None:
            assert debug_info.kmeans_responsibilities.shape[0] == n_spikes
            kmeans_labels = debug_info.kmeans_responsibilities.argmax(dim=1)
            kmeans_labels = kmeans_labels.cpu()
            colors = self.colors[kmeans_labels]
        else:
            kmeans_labels = torch.zeros(n_spikes, dtype=torch.long)
            colors = np.broadcast_to(np.array([self.bail_color]), (n_spikes,))

        # compute amplitudes
        if debug_info.split_data is not None:
            x = debug_info.split_data.x.view(
                n_spikes, -1, mix_data.tmm.neighb_cov.max_nc_obs
            )
            t = mix_data.train_times[debug_info.split_data.indices.cpu()]
            amps = x.square().sum(dim=1).amax(dim=1).sqrt_().numpy(force=True)
        else:
            t = amps = None

        # compute pca embedding of kmeans_x
        # TODO: also embed split_model means and maybe final model?
        if debug_info.kmeans_x is not None:
            assert debug_info.kmeans_x.shape[0] == n_spikes
            assert debug_info.kmeans_chans is not None
            kmean = debug_info.kmeans_x.mean(0)
            loadings, components, *_ = spiketorch.svd_lowrank_helper(
                x=debug_info.kmeans_x,
                rank=2,
                with_loadings=True,
                M=kmean[None].broadcast_to(debug_info.kmeans_x.shape),
            )
            assert loadings is not None
            loadings = loadings.numpy(force=True)
        else:
            loadings = None

        # distances
        if debug_info.split_model is not None:
            dists = debug_info.split_model.unit_distance_matrix()
            dists = dists.numpy(force=True)
        else:
            dists = None

        # mean reconstructions near my main channel
        mean_chans = mix_data.chans_in_radius(unit_id, radius=self.vis_radius)
        if debug_info.split_model is not None:
            means = debug_info.split_model.b.means
            means = means.view(means.shape[0], -1, mix_data.tmm.neighb_cov.n_channels)
            means = means[:, :, mean_chans]
        else:
            means = None

        return (
            txt,
            colors,
            t,
            amps,
            loadings,
            dists,
            means,
            mean_chans,
        )

    def draw(self, panel, mix_data: MixtureVisData, unit_id: int):
        c = self.compute(mix_data, unit_id)
        txt, colors, t, amps, loadings, dists, means, mean_chans = c

        # layout
        amp_info_row, mean_row, dist_pc_row = panel.subfigures(nrows=3)
        ax_amp, ax_info = amp_info_row.subplots(ncols=2)
        ax_mean = mean_row.subplots()
        fig_dist, fig_pc = dist_pc_row.subfigures(ncols=2)
        ax_pc = fig_pc.subplots()

        # unit dist matrix / pair mask
        if dists is not None:
            distance_matrix_dendro(
                panel=fig_dist,
                distances=dists,
                show_unit_labels=True,
                vmax=mix_data.tmm.max_distance,
                image_cmap=self.dist_cmap,
                show_values=True,
            )
        else:
            ax_dist = fig_dist.subplots()
            ax_dist.axis("off")
            ax_dist.text(0, 0, "no dists", fontsize=5)

        # merge res text info
        ax_info.axis("off")
        ax_info.text(0, 0, txt, fontsize=6)

        # pc scatter
        if loadings is not None:
            ax_pc.scatter(*loadings.T, colors=colors, s=5, lw=0, alpha=0.5)
            ax_pc.grid()
        else:
            ax_pc.axis("off")
            ax_pc.text(0, 0, "no kmeans pca", fontsize=5)

        # recon mean view
        if means is not None:
            chans = mean_chans[None].broadcast_to(len(means), *mean_chans.shape)
            geomplot(
                waveforms=means,
                channels=chans.numpy(force=True),
                geom=mix_data.prgeom,
                show_zero=False,
                colors=self.colors[: len(means)],
                zlim=None,
                ax=ax_mean,
                subar=True,
                annotate_z=True,
            )
        else:
            ax_mean.axis("off")
            ax_mean.text(0, 0, "no subunit means", fontsize=5)

        # amp view
        if amps is not None:
            ax_amp.scatter(t, amps, colors=colors, s=5, lw=0, alpha=0.5)
            ax_amp.grid()
            ax_amp.set_xlabel("t (s)")
            ax_amp.set_ylabel("feat maxchan norm")
        else:
            ax_amp.axis("off")
            ax_amp.text(0, 0, "no amps", fontsize=5)


figsize = (15, 8)
default_mixture_plots = (
    TextInfo(),
    ISIHistogram(),
    ChansHeatmap(),
    LikelihoodsView(),
    NeighborMeans(),
    NeighborDistances(),
    MergeView(),
    MeanView(),
    CovarianceView(),
    SplitView(),
)


def fit_mixture_and_visualize_all_components(
    *,
    sorting: DARTsortSorting,
    motion_est,
    refinement_cfg: RefinementConfig = default_refinement_cfg,
    computation_cfg: ComputationConfig | None = None,
    save_folder: str | Path,
    plots=default_mixture_plots,
    max_height=9,
    figsize=figsize,
    hspace=0.01,
    dpi=200,
    image_ext="png",
    n_jobs=0,
    show_progress=True,
    overwrite=False,
    unit_ids=None,
    n_units=None,
    seed=0,
    **other_global_params,
):
    computation_cfg = ensure_computation_config(computation_cfg)
    save_folder = resolve_path(save_folder)
    if unit_ids is None:
        unit_ids = sorting.unit_ids
    if n_units is not None and n_units < len(unit_ids):
        rg = np.random.default_rng(seed)
        unit_ids = rg.choice(unit_ids, size=n_units, replace=False)
    if not overwrite and all_summaries_done(unit_ids, save_folder, ext=image_ext):
        return
    mix_data = fit_mixture_for_vis(
        sorting=sorting,
        motion_est=motion_est,
        refinement_cfg=refinement_cfg,
        computation_cfg=computation_cfg,
    )
    return make_mixture_summaries(
        mix_data=mix_data,
        save_folder=save_folder,
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
        n_units=n_units,
        seed=seed,
        **other_global_params,
    )


def fit_mixture_for_vis(
    *,
    sorting: DARTsortSorting,
    motion_est,
    refinement_cfg: RefinementConfig = default_refinement_cfg,
    computation_cfg: ComputationConfig | None = None,
) -> MixtureVisData:
    # run model to convergence and soft assign train/full sets
    mix_data = instantiate_and_bootstrap_tmm(
        sorting=sorting,
        motion_est=motion_est,
        refinement_cfg=refinement_cfg,
        computation_cfg=computation_cfg,
    )
    mix_data.tmm.em(mix_data.train_data)
    train_scores = mix_data.tmm.soft_assign(
        data=mix_data.train_data,
        needs_bootstrap=False,
        full_proposal_view=False,
    )
    train_labels = labels_from_scores(train_scores)
    full_scores = mix_data.tmm.soft_assign(
        data=mix_data.full_data,
        needs_bootstrap=False,
        full_proposal_view=True,
    )
    full_labels = labels_from_scores(full_scores)
    if mix_data.val_data is None:
        eval_scores = train_scores
    else:
        eval_scores = mix_data.tmm.soft_assign(
            data=mix_data.val_data,
            needs_bootstrap=False,
            full_proposal_view=True,
        )

    dists = mix_data.tmm.unit_distance_matrix().cpu().clone()
    dists.diagonal().fill_(torch.inf)

    tpca = get_tpca(sorting)
    assert tpca is not None
    assert isinstance(tpca, TemporalPCA)

    return MixtureVisData(
        tmm=mix_data.tmm,
        train_data=mix_data.train_data,
        val_data=mix_data.val_data,
        full_data=mix_data.full_data,
        sorting=sorting,
        train_scores=train_scores,
        full_scores=full_scores,
        eval_scores=eval_scores,
        train_times=sorting.times_seconds[mix_data.train_ixs],  # type: ignore
        train_labels=train_labels,
        full_labels=full_labels,
        tpca=tpca,
        inf_diag_unit_distance_matrix=dists,
        prgeom=mix_data.tmm.neighb_cov.prgeom.numpy(force=True),
    )


def make_mixture_component_summary(
    mix_data: MixtureVisData,
    unit_id: int,
    plots=default_mixture_plots,
    max_height=9,
    figsize=figsize,
    hspace=0.01,
    figure=None,
    **other_global_params,
):
    for p in plots:
        p.notify_global_params(**other_global_params)
    figure = flow_layout(
        plots,
        max_height=max_height,
        figsize=figsize,
        hspace=hspace,
        figure=figure,
        mix_data=mix_data,
        unit_id=unit_id,
    )
    return figure


def make_mixture_summaries(
    mix_data: MixtureVisData,
    save_folder: str | Path,
    plots=default_mixture_plots,
    max_height=9,
    figsize=figsize,
    hspace=0.01,
    dpi=200,
    image_ext="png",
    n_jobs=0,
    show_progress=True,
    overwrite=False,
    unit_ids=None,
    n_units=None,
    seed=0,
    **other_global_params,
):
    save_folder = resolve_path(save_folder)
    if unit_ids is None:
        unit_ids = mix_data.tmm.unit_ids.numpy(force=True).tolist()
    if n_units is not None and n_units < len(unit_ids):
        rg = np.random.default_rng(seed)
        unit_ids = rg.choice(unit_ids, size=n_units, replace=False)
    if not overwrite and all_summaries_done(unit_ids, save_folder, ext=image_ext):
        return

    assert hasattr(mix_data, "log_liks")

    save_folder.mkdir(exist_ok=True, parents=True)

    global_params = dict(**other_global_params)

    n_jobs, Executor, context = get_pool(n_jobs, cls=CloudpicklePoolExecutor)  # type: ignore

    initargs = (
        mix_data,
        plots,
        max_height,
        figsize,
        hspace,
        dpi,
        save_folder,
        image_ext,
        overwrite,
        global_params,
        plt.rcParams,
    )
    # if ispar and not use_threads:
    #     initargs = (cloudpickle.dumps(initargs),)
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
        mix_data,
        plots,
        max_height,
        figsize,
        hspace,
        dpi,
        save_folder,
        image_ext,
        overwrite,
        global_params,
        rc,
    ):
        self.mix_data = mix_data
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
    _summary_job_context = SummaryJobContext(*args)


def _summary_job(unit_id):
    global _summary_job_context
    assert _summary_job_context is not None
    tmp_out = None
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
            dpi=_summary_job_context.dpi,
        )
        make_mixture_component_summary(
            _summary_job_context.mix_data,
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
        if tmp_out is not None and tmp_out.exists():
            tmp_out.unlink()
