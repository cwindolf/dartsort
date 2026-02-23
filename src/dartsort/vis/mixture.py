import math
from pathlib import Path
from typing import Iterable, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from dredge.motion_util import MotionEstimate
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.csgraph import connected_components
from tqdm.auto import tqdm

from ..clustering.cluster_util import maximal_leaf_groups
from ..clustering.gmm.mixture import (
    NeighborhoodLUT,
    Scores,
    StreamingSpikeData,
    TruncatedMixtureModel,
    TruncatedSpikeData,
    instantiate_and_bootstrap_tmm,
    labels_from_scores,
    labels_from_scores_,
    run_merge,
    run_split,
    try_kmeans,
)
from ..transform import TemporalPCA
from ..util import spiketorch
from ..util.data_util import DARTsortSorting, get_tpca, resolve_path
from ..util.internal_config import (
    ComputationConfig,
    InterpolationParams,
    RefinementConfig,
    default_refinement_cfg,
)
from ..util.interpolation_util import (
    NeighborhoodFiller,
    StableFeaturesInterpolator,
    pad_geom,
)
from ..util.job_util import ensure_computation_config
from ..util.multiprocessing_util import CloudpicklePoolExecutor, get_pool
from ..util.py_util import databag
from .analysis_plots import distance_matrix_dendro
from .colors import glasbey1024
from .layout import BasePlot, flow_layout
from .waveforms import geomplot


@databag
class MixtureVisData:
    """Collection passed around to the plots in this file, with helper methods for vis."""

    tmm: TruncatedMixtureModel
    train_data: TruncatedSpikeData
    val_data: TruncatedSpikeData | None
    full_data: StreamingSpikeData
    sorting: DARTsortSorting
    motion_est: MotionEstimate | None
    train_scores: Scores
    full_scores: Scores
    eval_scores: Scores
    eval_labels: torch.Tensor
    train_times: np.ndarray
    train_labels: np.ndarray
    train_ixs: np.ndarray
    val_ixs: np.ndarray | None
    full_labels: np.ndarray
    tpca: TemporalPCA
    inf_diag_unit_distance_matrix: torch.Tensor
    prgeom: np.ndarray

    @property
    def times_seconds(self):
        return self.sorting.times_seconds  # type: ignore

    def to_sorting(self) -> DARTsortSorting:
        return self.sorting.ephemeral_replace(labels=self.full_labels)

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
            dists = np.ascontiguousarray(dists[::-1])
            neighbors = np.ascontiguousarray(neighbors[::-1])
        return dists, neighbors

    def reconstruct_flat(self, features: torch.Tensor) -> np.ndarray:
        frank = self.tmm.neighb_cov.feat_rank
        single = features.ndim == 1
        if single:
            features = features[None]
        assert features.ndim == 2
        n = features.shape[0]
        features = features.view(n, frank, -1)
        features = features.to(self.tpca.b.components.device)
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

    def full_lut_scores(self, show_progress=2, n_candidates=None):
        lut0 = self.tmm.lut
        lut1 = torch.ones_like(lut0.lut)
        unit_ids, neighb_ids = lut1.nonzero(as_tuple=True)
        lut1[unit_ids, neighb_ids] = torch.arange(unit_ids.shape[0]).to(lut1)
        new_lut = NeighborhoodLUT(unit_ids=unit_ids, neighb_ids=neighb_ids, lut=lut1)
        self.tmm.update_lut(new_lut, no_parameter_changes=True)
        # old_nc = self.full_data.n_candidates
        sc = self.tmm.soft_assign(
            data=self.full_data,
            needs_bootstrap=False,
            full_proposal_view=True,
            show_progress=show_progress,
        )
        return sc


class MixtureComponentPlot(BasePlot):
    width = 1
    height = 1
    kind = "mixture"

    def draw(self, panel, mix_data: MixtureVisData, unit_id: int):
        raise NotImplementedError


class TextInfo(MixtureComponentPlot):
    kind = "small"
    height = 0.5

    def draw(self, panel, mix_data: MixtureVisData, unit_id: int):
        axis = panel.subplots()
        axis.axis("off")

        ntot = (mix_data.full_labels == unit_id).sum()
        ntrain = (mix_data.train_labels == unit_id).sum()

        msg = f"{_bold(f'unit {unit_id}')}\n"
        msg += f"n total: {ntot}\n"
        msg += f"n train: {ntrain}\n"

        axis.text(
            0.025,
            0.5,
            msg,
            fontsize="small",
            ha="left",
            va="center",
            transform=axis.transAxes,
        )


class ISIHistogram(MixtureComponentPlot):
    kind = "small"
    height = 1.0

    def __init__(self, bin_ms=0.1, max_ms=5):
        self.bin_ms = bin_ms
        self.max_ms = max_ms

    def draw(self, panel, mix_data: MixtureVisData, unit_id: int):
        axis = panel.subplots()
        inu_full, times_s = mix_data.inu_and_times_full(unit_id)
        dt_ms = np.diff(times_s) * 1000
        bin_edges = np.arange(
            -0.5 * self.bin_ms, self.max_ms + self.bin_ms + 1e-5, self.bin_ms
        )
        counts, _ = np.histogram(dt_ms, bin_edges)
        axis.stairs(counts, bin_edges, color=glasbey1024[unit_id], fill=True)
        axis.set_xlabel(f"isi (ms, {dt_ms.size + 1} tot. sp.)")
        axis.set_ylabel("count")


class ChansHeatmap(MixtureComponentPlot):
    kind = "small"
    height = 1.5

    def __init__(self, cmap="magma", snr_cmap="viridis"):
        self.cmap = plt.get_cmap(cmap)
        self.snr_cmap = plt.get_cmap(snr_cmap)

    def draw(self, panel, mix_data: MixtureVisData, unit_id: int):
        inu_train, chans = mix_data.train_inds_and_chans(unit_id)
        chans = chans[chans < mix_data.tmm.neighb_cov.n_channels]
        uchans, counts = np.unique(chans, return_counts=True)

        mean = mix_data.tmm.b.means[unit_id].view(
            -1, mix_data.tmm.neighb_cov.n_channels
        )
        ptp = spiketorch.ptp(mean, dim=0).cpu()
        (support,) = ptp[uchans].nonzero(as_tuple=True)
        support = uchans[support]
        ptp = ptp[support]

        ax = panel.subplots()
        xy = mix_data.tmm.neighb_cov.prgeom.numpy(force=True)
        ax.scatter(*xy[support].T, c=ptp, lw=2, s=5)
        s = ax.scatter(*xy[uchans].T, c=counts, lw=0, cmap=self.cmap, s=5)
        plt.colorbar(s, ax=ax, shrink=0.3, label="chan count", pad=0.01)
        ax.scatter(*xy[support[ptp.argmax()]].T, color="gold", lw=1, fc="none", s=5)
        ax.set_title(
            f"chans:{chans.min().item()}-{chans.max().item()}", fontsize="small"
        )


class LikelihoodsView(MixtureComponentPlot):
    def __init__(self, viol_ms=1.0, layout="vert"):
        self.viol_ms = viol_ms
        self.layout = layout
        if layout == "vert":
            self.kind = "small"
            self.width = 1
            self.height = 4
            self.ncols = 1
            self.nrows = 3
            self.width_ratios = [1]
        elif layout == "horz":
            self.kind = "block"
            self.width = 2
            self.height = 1
            self.ncols = 3
            self.nrows = 1
            self.width_ratios = [3, 2, 1]
        else:
            assert False

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
        my_min = my_ll.min()
        my_max = my_ll.max()
        noise_min = noise_ll.min()
        noise_max = noise_ll.max()
        mn = min(my_min, noise_min)
        mx = max(my_max, noise_max)
        assert math.isfinite(mn)
        return my_ll, noise_ll, t, viol, mn, mx

    def draw(self, panel, mix_data: MixtureVisData, unit_id: int):
        my_ll, noise_ll, t, viol, mn, mx = self.compute(mix_data, unit_id)

        ax_time, ax_noise, ax_hist = panel.subplots(
            ncols=self.ncols,
            nrows=self.nrows,
            width_ratios=self.width_ratios,
            sharey=True,
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
            frameon=False,
            borderpad=0.1,
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

    def __init__(self, count=5, cmap="plasma"):
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
            vmax=mix_data.tmm.p.split_max_distance,
            image_cmap=self.cmap,
            show_values=True,
        )
        panel.suptitle(f"{mix_data.tmm.p.distance_kind} dists", fontsize="small")


class MergeView(MixtureComponentPlot):
    kind = "block"
    width = 2
    height = 2.5

    def __init__(self):
        pass

    def compute(self, mix_data: MixtureVisData, unit_id: int):
        # -- get actual group used during merge
        # start by getting local distance matrix D for neighbors within merge distance
        d0 = mix_data.inf_diag_unit_distance_matrix[unit_id]
        (neighbors,) = (d0 < mix_data.tmm.p.merge_max_distance).nonzero(as_tuple=True)
        me = neighbors.new_full((1,), unit_id)
        neighbors = torch.cat([neighbors, me]).sort().values
        D = mix_data.inf_diag_unit_distance_matrix[neighbors][:, neighbors].clone()

        # find my complete linkage cluster within D
        D = D.fill_diagonal_(0.0).numpy(force=True)
        if D.shape[0] > 1:
            pd = D[np.triu_indices(D.shape[0], k=1)]
            Z = linkage(pd, method="complete")
            groups = maximal_leaf_groups(
                Z,
                distances=D,
                max_distance=mix_data.tmm.p.merge_max_distance,
                max_group_size=mix_data.tmm.p.max_group_size,
            )
            groups = [g for g in groups if unit_id in neighbors[list(g)].tolist()]
            assert len(groups) == 1
            group_ix = list(groups[0])
            group = neighbors[group_ix]
        else:
            group_ix = np.arange(neighbors.shape[0])
            group = neighbors
        del neighbors

        # get pair mask
        D = D[group_ix][:, group_ix]
        pair_mask = torch.asarray(D < mix_data.tmm.p.merge_max_distance)

        # run it
        group_res = mix_data.tmm.try_merge_group(
            group=torch.asarray(group),
            train_data=mix_data.train_data,
            eval_data=mix_data.val_data,
            eval_scores=mix_data.eval_scores,
            train_labels=torch.asarray(mix_data.train_labels),
            eval_labels=mix_data.eval_labels,
            pair_mask=pair_mask,
            apply_adj_mask=True,
            debug=True,
        )

        return group, D, pair_mask, group_res

    def draw(self, panel, mix_data: MixtureVisData, unit_id: int):
        neighbors, dist, pair_mask, group_res = self.compute(mix_data, unit_id)
        panel_mask, panel_info = panel.subfigures(nrows=2, height_ratios=[2, 0.5])
        distance_matrix_dendro(
            panel_mask,
            pair_mask,
            show_values_from=dist,
            unit_ids=neighbors,
            dendrogram_linkage=None,
            show_unit_labels=True,
            image_cmap="RdGy_r",
            show_values=True,
            with_colorbar=False,
            value_color="w",
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
        ax_info.text(
            0.2, 0.5, msg, fontsize="small", va="center", transform=ax_info.transAxes
        )
        panel.suptitle("merge pair mask", fontsize="small")


class MeanView(MixtureComponentPlot):
    def __init__(
        self,
        n_waveforms_show=128,
        alpha=0.25,
        show_zero=True,
        mini=False,
        mini_rad=41.0,
    ):
        self.n_waveforms_show = n_waveforms_show
        self.alpha = alpha
        self.show_zero = show_zero
        self.mini = mini
        self.mini_rad = mini_rad
        if mini:
            self.kind = "tall"
            self.width = 2.5
            self.height = 2
        else:
            self.kind = "bigblock"
            self.width = 3
            self.height = 4

    def compute(self, mix_data: MixtureVisData, unit_id: int):
        mean = mix_data.tmm.b.means[unit_id]
        mean_recon = mix_data.reconstruct_flat(mean)
        mean_recon = mean_recon.reshape(1, -1, mix_data.tmm.neighb_cov.n_channels)
        inu_train, wchans, features, waveforms = mix_data.random_train_waveforms(
            unit_id=unit_id, count=self.n_waveforms_show
        )
        if self.mini:
            targchans = mix_data.chans_in_radius(unit_id, radius=self.mini_rad)
            wchans = torch.asarray(wchans).to(targchans)
            vchans = torch.isin(wchans, targchans)
            vii, vcc = vchans.cpu().nonzero(as_tuple=True)
            wchans = wchans[vii, vcc].cpu()
            waveforms = waveforms[vii, :, vcc]
            wchans = wchans[:, None]
            assert wchans.ndim == 2
            waveforms = waveforms[:, :, None]
            assert waveforms.ndim == 3
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
            linewidth=0.5,
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
    height = 3

    def __init__(
        self, n_waveforms_show=128, neigs=16, cov_vert=False, reference_cov="interp"
    ):
        self.n_waveforms_show = n_waveforms_show
        self.colors = dict(emp="k", interp="gray", noise="r", signal="g", model="b")
        self.neigs = neigs
        self.cov_vert = cov_vert
        self.reference_cov = reference_cov

    def compute(self, mix_data: MixtureVisData, unit_id: int):
        # load data
        inu_train, wchans, features, waveforms = mix_data.random_train_waveforms(
            unit_id=unit_id, count=self.n_waveforms_show
        )

        # pick channels and put data there
        chan_set = torch.asarray(wchans).unique().to(device=mix_data.tmm.b.means.device)
        chan_set = chan_set[chan_set < mix_data.tmm.neighb_cov.n_channels]
        features = features.reshape(
            features.shape[0], -1, mix_data.tmm.neighb_cov.max_nc_obs
        )
        features = torch.asarray(features)
        # -- with nans
        feat_nan = features.new_full(
            (*features.shape[:2], mix_data.tmm.neighb_cov.n_channels + 1), torch.nan
        )
        ix = torch.asarray(wchans)
        ix = ix[:, None, :].broadcast_to(features.shape)
        feat_nan.scatter_(
            dim=2,
            index=ix,
            src=features,
        )
        feat_nan = feat_nan[:, :, chan_set.cpu()]
        feat_nan = feat_nan.view(feat_nan.shape[0], -1)
        # -- with interp
        feat_interp = mix_data.tmm.erp.interp_to_chans(
            waveforms=features.to(mix_data.tmm.b.means),
            neighborhood_ids=mix_data.train_data.neighborhood_ids[inu_train],
            target_channels=chan_set,
        )
        feat_interp = feat_interp.view(feat_interp.shape[0], -1)

        # empirical covariance
        mean_nan = torch.nanmean(feat_nan, dim=0)
        feat_nan -= mean_nan
        cov_nan = cast(torch.Tensor, spiketorch.nancov(feat_nan, correction=0))

        # interp covariance
        cov_interp = torch.cov(feat_interp.T)

        # noise covariance
        cov_noise = mix_data.tmm.noise.marginal_covariance(channels=chan_set).to_dense()
        assert cov_noise.shape == cov_interp.shape

        # signal covariance
        if mix_data.tmm.signal_rank:
            basis = mix_data.tmm.b.bases[unit_id]
            basis = basis.view(
                mix_data.tmm.signal_rank, -1, mix_data.tmm.neighb_cov.n_channels
            )
            basis = basis[:, :, chan_set].view(mix_data.tmm.signal_rank, -1)
            basis = basis * mix_data.tmm.p.latent_prior_std
            cov_signal = basis.T @ basis
            assert cov_signal.shape == cov_noise.shape
        else:
            cov_signal = torch.zeros_like(cov_noise)

        # model covariance
        cov_model = cov_noise + cov_signal

        # np-ify
        cov_nan = cov_nan.numpy(force=True)
        cov_interp = cov_interp.numpy(force=True)
        cov_noise = cov_noise.numpy(force=True)
        cov_signal = cov_signal.numpy(force=True)
        cov_model = cov_model.numpy(force=True)

        # spectra: empirical, noise, sginal, model
        ev_nan = np.linalg.eigvalsh(cov_nan)[::-1]
        ev_interp = np.linalg.eigvalsh(cov_interp)[::-1]
        ev_noise = np.linalg.eigvalsh(cov_noise)[::-1]
        ev_signal = np.linalg.eigvalsh(cov_signal)[::-1]
        ev_model = np.linalg.eigvalsh(cov_model)[::-1]

        return {
            "emp": (cov_nan, ev_nan),
            "interp": (cov_interp, ev_interp),
            "noise": (cov_noise, ev_noise),
            "signal": (cov_signal, ev_signal),
            "model": (cov_model, ev_model),
        }

    def draw(self, panel, mix_data: MixtureVisData, unit_id: int):
        stats = self.compute(mix_data, unit_id)
        cov_emp = stats[self.reference_cov][0]
        vm = np.percentile(np.abs(cov_emp), 98)

        toph = 2 + int(self.cov_vert)
        top, bottom = panel.subfigures(nrows=2, height_ratios=[toph, 1])
        if self.cov_vert:
            axes_top = top.subplots(ncols=2, nrows=len(stats))
        else:
            axes_top = top.subplots(ncols=len(stats), nrows=2)
            axes_top = axes_top.T
        ax_bottom = bottom.subplots()

        cov_kw = dict(
            aspect=1.0, interpolation="none", vmin=-vm, vmax=vm, cmap="seismic"
        )
        res_kw = cov_kw | dict(vmin=-vm, vmax=vm)

        for row_top, (name, (cov, ev)) in zip(axes_top, stats.items()):
            c = self.colors[name]
            if self.cov_vert:
                row_top[0].set_ylabel(name, color=c)
            else:
                row_top[0].set_title(name, color=c, fontsize=6)
            ima = row_top[0].imshow(cov, **cov_kw)
            imb = row_top[1].imshow(cov_emp - cov, **res_kw)
            ax_bottom.plot(ev[: self.neigs], color=c, label=name)
        if not self.cov_vert:
            plt.colorbar(ima, ax=row_top[0], shrink=0.3)  # type: ignore
            plt.colorbar(imb, ax=row_top[1], shrink=0.3)  # type: ignore

        if self.cov_vert:
            axes_top[0, 0].set_title("cov", fontsize="small")
            axes_top[0, 1].set_title(
                f"{self.reference_cov} - cov",
                fontsize="small",
                color=self.colors[self.reference_cov],
            )
        else:
            axes_top[0, 0].set_ylabel("cov", fontsize="small")
            axes_top[0, 1].set_ylabel(
                f"{self.reference_cov} - cov",
                fontsize="small",
                color=self.colors[self.reference_cov],
            )
        for ax in axes_top.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        ax_bottom.grid(which="both")
        ax_bottom.legend(
            borderpad=0.1,
            borderaxespad=0.1,
            handletextpad=0.3,
            handlelength=1.0,
            loc="upper right",
            ncols=2,
        )


class SplitView(MixtureComponentPlot):
    kind = "tall"
    width = 2.5
    height = 5

    def __init__(
        self,
        colors=["r", "g", "b", "darkorange", "darkviolet", "mediumturquoise"],
        bail_color="k",
        vis_radius=50.0,
        dist_cmap="plasma",
    ):
        self.colors = np.array(colors)
        self.bail_color = bail_color
        self.vis_radius = vis_radius
        self.dist_cmap = plt.get_cmap(dist_cmap)

    def compute(self, mix_data: MixtureVisData, unit_id: int):
        # my group...
        if mix_data.tmm.p.split_friend_distance:
            _, friends = mix_data.friends(
                unit_id, count=mix_data.tmm.p.max_group_size, me_last=False
            )
            friends = torch.as_tensor(friends)
            D = mix_data.inf_diag_unit_distance_matrix[friends][:, friends].clone()
            D.fill_diagonal_(0.0)
            pd = D.numpy()[np.triu_indices(D.shape[0], k=1)]
            Z = linkage(pd, method="complete")
            fcl = fcluster(
                Z, mix_data.tmm.p.split_friend_distance, criterion="distance"
            )
            group = torch.as_tensor(friends[fcl == fcl[0]])
        else:
            group = torch.tensor([unit_id])

        split_res, debug_info = mix_data.tmm.split_group(
            group=group,
            train_data=mix_data.train_data,
            eval_data=mix_data.val_data,
            eval_scores=mix_data.eval_scores,
            train_labels=torch.asarray(mix_data.train_labels),
            eval_labels=mix_data.eval_labels,
            debug=True,
        )
        assert debug_info is not None
        if debug_info.split_data is not None:
            orig_labels = mix_data.train_labels[debug_info.split_data.indices.cpu()]
        else:
            orig_labels = None

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

        # compute channels by kmeans label
        if debug_info.kmeans_responsibilities is not None:
            assert debug_info.split_data is not None
            chans_by_km = {}
            for l in kmeans_labels.unique():
                neighbs_l = debug_info.split_data.neighborhood_ids[kmeans_labels == l]
                chans_l = mix_data.tmm.neighb_cov.obs_ix[neighbs_l]
                chans_l = chans_l[chans_l < mix_data.tmm.neighb_cov.n_channels]
                chans_by_km[l.item()] = chans_l.numpy(force=True)
        else:
            chans_by_km = None

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
            means = means.view(means.shape[0], -1)
            means = mix_data.reconstruct_flat(means)
        else:
            means = None

        # compute text info
        txt = f"{_bold('ids')}: {group.cpu().tolist()} "
        if debug_info.bailed:
            txt += f"bail: {debug_info.bail_reason}\n"
        if debug_info.merge_res is not None:
            pstr = ",".join(map(str, debug_info.merge_res.grouping.group_ids.tolist()))
            txt += (
                f"{_bold('part')}: {pstr} "
                f"{_bold('imp')}: {debug_info.merge_res.improvement:.3f}\n"
            )
        else:
            txt += f"no allowed partitions\n"
        if debug_info.split_model is not None:
            sprops = debug_info.split_model.b.log_proportions.cpu()
            sprops = (
                sprops - mix_data.tmm.b.log_proportions[group].logsumexp(dim=0).cpu()
            )
            propstr = "+".join(f"{p:0.2f}" for p in sprops.exp())
            txt += f"kmprops: {propstr}={sprops.exp().sum():.2f}\n"
        if debug_info.merge_res is not None:
            mresprops = debug_info.merge_res.sub_proportions
            propstr = "+".join(f"{p:0.2f}" for p in mresprops)
            txt += f"mresprops: {propstr}={mresprops.sum():.2f}\n"
        if debug_info.kmeans_responsibilities is not None:
            kcountstr = ",".join(
                str(p.item()) for p in kmeans_labels.unique(return_counts=True)[1]
            )
            txt += f"kmeans counts: {kcountstr}\n"
        if split_res is not None:
            propstr = "+".join(f"{p:0.3f}" for p in split_res.sub_proportions.cpu())
            countstr = ",".join(
                str(p.item())
                for p in split_res.train_assignments.unique(return_counts=True)[1]
            )
            txt += f"props: {propstr}={split_res.sub_proportions.cpu().sum():.2f}\n"
            txt += f"counts: {countstr}\n"
        if orig_labels is not None and group.numel() > 1 and split_res is not None:
            cstrs = []
            for uid in group.tolist():
                myl = split_res.train_assignments[orig_labels == uid]
                uu, cc = myl.unique(return_counts=True)
                cc = ",".join(f"{int(uuu)}:{int(ccc)}" for uuu, ccc in zip(uu, cc))
                cstrs.append(f"{uid}->[{cc}]")
            cstr = "orig: " + "\n      ".join(cstrs)
            txt += cstr + "\n"

            cstrs = []
            for uid in range(split_res.n_split):
                myl = orig_labels[split_res.train_assignments.cpu() == uid]
                uu, cc = np.unique(myl, return_counts=True)
                cc = ",".join(f"{int(uuu)}:{int(ccc)}" for uuu, ccc in zip(uu, cc))
                cstrs.append(f"{uid}->[{cc}]")
            cstr = "new:  " + "\n      ".join(cstrs)
            txt += cstr + "\n"
        txt = txt.rstrip()

        if debug_info.split_data is not None:
            split_inds = debug_info.split_data.indices.cpu()
            tw = debug_info.split_data.duties
            if tw is not None:
                tw = tw.numpy(force=True)
        else:
            tw = split_inds = None

        return (
            txt,
            colors,
            t,
            amps,
            loadings,
            dists,
            means,
            mean_chans,
            chans_by_km,
            split_inds,
            tw,
        )

    def draw(self, panel, mix_data: MixtureVisData, unit_id: int):
        c = self.compute(mix_data, unit_id)
        (
            txt,
            colors,
            t,
            amps,
            loadings,
            dists,
            means,
            mean_chans,
            chans_by_km,
            split_inds,
            tw,
        ) = c

        # layout
        amp_info_row, mean_row, dist_pc_row = panel.subfigures(
            nrows=3, height_ratios=[1, 1, 1.5]
        )
        ax_amp, ax_info = amp_info_row.subplots(ncols=2)
        ax_mean = mean_row.subplots()
        fig_dist_chans, fig_pc = dist_pc_row.subfigures(ncols=2)
        ax_pc = fig_pc.subplots()
        fig_dist, fig_chans = fig_dist_chans.subfigures(nrows=2, height_ratios=[3, 2])
        ax_chans = fig_chans.subplots()

        # unit dist matrix / pair mask
        if dists is not None:
            dists_inf = dists.copy()
            dists_inf[dists > mix_data.tmm.p.split_max_distance] = np.inf
            distance_matrix_dendro(
                panel=fig_dist,
                distances=dists_inf,
                show_values_from=dists,
                show_unit_labels=True,
                vmax=mix_data.tmm.p.split_max_distance,
                image_cmap=self.dist_cmap,
                show_values=True,
                label_colors=self.colors,
                with_colorbar=False,
            )
        else:
            ax_dist = fig_dist.subplots()
            ax_dist.axis("off")
            ax_dist.text(0, 0, "no dists", fontsize=5)

        if chans_by_km is not None:
            low = min(c.min().item() for c in chans_by_km.values())
            hi = max(c.max().item() for c in chans_by_km.values())
            bins = np.arange(low - 0.5, hi + 0.6, step=1.0)
            ax_chans.hist(
                list(chans_by_km.values()),
                stacked=True,
                color=self.colors[list(chans_by_km.keys())],
                bins=bins,
            )

        # merge res text info
        ax_info.axis("off")
        ax_info.text(
            0.2,
            0.5,
            txt,
            fontsize="small",
            va="center",
            transform=ax_info.transAxes,
        )

        # pc scatter
        if loadings is not None and tw is not None:
            s = ax_pc.scatter(*loadings.T, c=tw, s=5, lw=0, alpha=1)
            plt.colorbar(s, ax=ax_pc, shrink=0.3)
            ax_pc.scatter(
                *loadings.T, edgecolors=colors, s=10, lw=0.5, facecolor="none", alpha=1
            )
            ax_pc.grid()
        elif loadings is not None and tw is not None:
            ax_pc.scatter(*loadings.T, c=colors, s=5, alpha=0.5)
            ax_pc.grid()
        else:
            ax_pc.axis("off")
            ax_pc.text(0, 0, "no kmeans pca", fontsize=5)

        # recon mean view
        ax_mean.axis("off")
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
            ax_mean.text(0, 0, "no subunit means", fontsize=5)

        # amp view
        if amps is not None:
            ax_amp.scatter(t, amps, c=colors, s=5, lw=0, alpha=0.5)
            ax_amp.grid()
            ax_amp.set_xlabel("t (s)")
            ax_amp.set_ylabel("feat maxchan norm")
        else:
            ax_amp.axis("off")
            ax_amp.text(0, 0, "no amps", fontsize=5)
        panel.suptitle("split result vis")


def default_mixture_plots():
    return (
        TextInfo(),
        ISIHistogram(),
        ChansHeatmap(),
        LikelihoodsView(),
        NeighborMeans(),
        NeighborDistances(),
        MergeView(),
        MeanView(),
        CovarianceView(),
        MeanView(mini=True),
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
    figsize=(17, 10),
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
    em: bool = True,
    split: bool = False,
    merge: bool = False,
    both: bool = False,
) -> MixtureVisData:
    # run model to convergence and soft assign train/full sets
    mix_data = instantiate_and_bootstrap_tmm(
        sorting=sorting,
        motion_est=motion_est,
        refinement_cfg=refinement_cfg,
        computation_cfg=computation_cfg,
    )
    if em:
        mix_data.tmm.em(mix_data.train_data)
    if split or both:
        run_split(mix_data.tmm, mix_data.train_data, mix_data.val_data, prog_level=1)
        mix_data.tmm.em(mix_data.train_data)
    if merge or both:
        run_merge(mix_data.tmm, mix_data.train_data, mix_data.val_data, prog_level=1)
        mix_data.tmm.em(mix_data.train_data)

    train_scores = mix_data.tmm.soft_assign(
        data=mix_data.train_data,
        needs_bootstrap=False,
        full_proposal_view=True,
    )
    train_labels = labels_from_scores_(train_scores)
    full_scores = mix_data.tmm.soft_assign(
        data=mix_data.full_data,
        needs_bootstrap=False,
        full_proposal_view=True,
    )
    full_labels = labels_from_scores(full_scores)
    if mix_data.val_data is None:
        eval_scores = train_scores
        eval_labels = train_labels
    else:
        eval_scores = mix_data.tmm.soft_assign(
            data=mix_data.val_data,
            needs_bootstrap=False,
            full_proposal_view=True,
        )
        eval_labels = labels_from_scores_(eval_scores)

    dists = mix_data.tmm.unit_distance_matrix().cpu().clone()
    dists.diagonal().fill_(torch.inf)

    tpca = get_tpca(sorting)
    assert tpca is not None
    assert isinstance(tpca, TemporalPCA)
    if isinstance(mix_data.train_ixs, slice):
        assert mix_data.train_ixs == slice(None)
        assert train_labels.shape == full_labels.shape
        train_ixs = np.arange(train_labels.shape[0])
    else:
        train_ixs = mix_data.train_ixs.numpy(force=True)

    assert not isinstance(mix_data.val_ixs, slice)
    if isinstance(mix_data.val_ixs, torch.Tensor):
        val_ixs = mix_data.val_ixs.numpy(force=True)
    else:
        val_ixs = None

    times_s = cast(np.ndarray, getattr(sorting, "times_seconds"))

    return MixtureVisData(
        tmm=mix_data.tmm,
        train_data=mix_data.train_data,
        val_data=mix_data.val_data,
        full_data=mix_data.full_data,
        sorting=sorting,
        motion_est=motion_est,
        train_scores=train_scores,
        full_scores=full_scores,
        eval_scores=eval_scores,
        train_times=times_s[mix_data.train_ixs],
        train_ixs=train_ixs,
        val_ixs=val_ixs,
        train_labels=train_labels.numpy(force=True),
        eval_labels=eval_labels,
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
    figsize=(17, 10),
    card_hspace=0.01,
    figure=None,
    **other_global_params,
):
    if callable(plots):
        plots = plots()
    plots = cast(Iterable[MixtureComponentPlot], plots)
    for p in plots:
        p.notify_global_params(**other_global_params)
    figure = flow_layout(
        plots,
        max_height=max_height,
        figsize=figsize,
        card_hspace=card_hspace,
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
    figsize=(17, 10),
    card_hspace=0.01,
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

    save_folder.mkdir(exist_ok=True, parents=True)
    global_params = dict(**other_global_params)
    n_jobs, Executor, context = get_pool(n_jobs, cls=CloudpicklePoolExecutor)  # type: ignore

    initargs = (
        mix_data,
        plots,
        max_height,
        figsize,
        card_hspace,
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
        card_hspace,
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
        self.card_hspace = card_hspace
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
            card_hspace=_summary_job_context.card_hspace,
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


# -- one-offs


def vis_split_interpolation(
    mix_data: MixtureVisData,
    unit_id: int,
    erp: NeighborhoodFiller | None = None,
    whiten: bool = True,
    figscale=5,
    n_per_group=8,
    seed=0,
    layout="horz",
    overlay=True,
):
    if erp is None:
        erp = mix_data.tmm.erp

    split_data = mix_data.train_data.dense_slice_by_unit(
        unit_id,
        gen=mix_data.tmm.rg,
        min_count=2 * mix_data.tmm.p.min_count,
        labels=None,
    )
    assert split_data is not None

    kmeans_responsibilities, kmeans_x, kmeans_chans = try_kmeans(
        data=split_data,
        k=mix_data.tmm.p.split_k,
        erp=erp,
        gen=mix_data.tmm.rg,
        feature_rank=mix_data.tmm.noise.rank,
        min_count=mix_data.tmm.p.min_count,
        min_channel_count=mix_data.tmm.p.min_channel_count,
        debug=True,
        whiten=whiten,
    )

    n_spikes = split_data.x.shape[0]
    assert kmeans_x is not None
    assert kmeans_x.shape[0] == n_spikes
    assert kmeans_chans is not None
    if kmeans_responsibilities is not None:
        assert kmeans_responsibilities.shape[0] == n_spikes
        kmeans_labels = kmeans_responsibilities.argmax(dim=1)
        kmeans_labels = kmeans_labels.cpu()
    else:
        kmeans_labels = torch.zeros(n_spikes, dtype=torch.long)

    vis_ix = []
    rg = np.random.default_rng(seed)
    ulabels = kmeans_labels.unique()
    for l in ulabels:
        (in_l,) = (kmeans_labels == l).nonzero(as_tuple=True)
        in_l = in_l.numpy(force=True)
        if in_l.size > n_per_group:
            in_l = rg.choice(in_l, size=n_per_group, replace=False)
        vis_ix.append(in_l)

    # interpolated waveforms
    interp_wfs = []
    for vix in vis_ix:
        f = kmeans_x[vix].reshape(len(vix), -1)
        f = mix_data.reconstruct_flat(f)
        interp_wfs.append(f)

    # original observed waveforms
    orig_wfs = []
    orig_chans = []
    for vix in vis_ix:
        f = (split_data.whitenedx if whiten else split_data.x)[vix]
        f = mix_data.reconstruct_flat(f)
        orig_wfs.append(f)
        orig_chans.append(
            split_data.neighborhoods.b.neighborhoods[split_data.neighborhood_ids[vix]]
        )

    # -- draw

    if layout == "horz":
        fig, axes = plt.subplots(
            nrows=2,
            ncols=ulabels.numel(),
            figsize=(figscale * ulabels.numel(), figscale * 2),
            layout="constrained",
        )
        axes = axes.T
    else:
        fig, axes = plt.subplots(
            ncols=2,
            nrows=ulabels.numel(),
            figsize=(figscale * 2, figscale * ulabels.numel()),
            layout="constrained",
        )
    maa = max([np.abs(w).max() for w in orig_wfs])

    for col, iwf, owf, ochans, c in zip(axes, interp_wfs, orig_wfs, orig_chans, "rgb"):
        geomplot(
            waveforms=iwf,
            channels=kmeans_chans[None].broadcast_to(iwf.shape[0], *kmeans_chans.shape),
            geom=mix_data.prgeom,
            show_zero=False,
            subar=False,
            ax=col[1],
            max_abs_amp=maa,
            linewidth=1,
            zlim=None,
        )
        for ax in col[: 1 + overlay]:
            geomplot(
                waveforms=owf,
                channels=ochans,
                geom=mix_data.prgeom,
                show_zero=True,
                subar=True,
                ax=ax,
                max_abs_amp=maa,
                color="k",
                linewidth=1,
                zlim=None,
            )
    for ax in axes.flat:
        ax.axis("off")

    return fig


def vis_obs_interpolation(
    mix_data: MixtureVisData,
    unit_id: int,
    count: int = 256,
    erp_params: dict[str, InterpolationParams] | None = None,
    figscale=5.0,
    disp_colormap="viridis",
    time_colormap="plasma",
):
    (trix,) = (mix_data.train_labels == unit_id).nonzero()
    if trix.size > count:
        trix = np.random.default_rng(0).choice(trix, size=count, replace=False)
        trix.sort()
    fix = mix_data.train_ixs[trix]
    n = len(trix)

    t_s = mix_data.train_times[trix]
    z = mix_data.sorting.slice_feature_by_name("point_source_localizations", fix)[:, 2]
    if mix_data.motion_est is None:
        disp = np.zeros_like(z)
    else:
        disp = mix_data.motion_est.disp_at_s(t_s, z)

    dev = mix_data.tmm.b.means.device
    sgeom = pad_geom(mix_data.sorting.geom, device=dev)  # type: ignore
    tgeom = torch.asarray(mix_data.prgeom).to(sgeom)
    channel_index = torch.asarray(mix_data.sorting.channel_index, device=dev)  # type: ignore
    erps = {}
    for k, ip in (erp_params or {}).items():
        erps[k] = StableFeaturesInterpolator(
            source_geom=sgeom,
            target_geom=tgeom,
            channel_index=channel_index,
            params=ip,
        )

    features = {"actual": mix_data.train_data.x[trix]}
    origf = mix_data.sorting.slice_feature_by_name(
        "collisioncleaned_tpca_features", fix
    )
    origf = torch.asarray(origf).to(sgeom)
    chans = torch.asarray(mix_data.sorting.channels[fix]).to(channel_index)
    shifts = torch.asarray(-disp).to(sgeom)
    tchans = mix_data.tmm.neighb_cov.obs_ix[mix_data.train_data.neighborhood_ids[trix]]
    for name, erp in (erps or {}).items():
        features[name] = erp.interp(
            features=origf,
            source_main_channels=chans,
            target_channels=tchans,
            source_shifts=shifts,
        )

    waveforms = {
        k: mix_data.reconstruct_flat(f.view(n, -1)) for k, f in features.items()
    }

    maa = origf.abs().nan_to_num_().max().cpu().item()
    dcolors = plt.get_cmap(disp_colormap)(spiketorch.minmax(disp))
    tcolors = plt.get_cmap(time_colormap)(spiketorch.minmax(t_s))

    ncols = 2
    nrows = len(features)
    fig = plt.figure(
        figsize=(ncols * figscale, (nrows + 0.5) * figscale),
        layout="constrained",
    )
    panels = fig.subfigures(
        nrows=nrows + 1,
        height_ratios=[1] + ([2] * nrows),
    )

    ax_top = panels[0].subplots()
    ax_top.scatter(t_s, disp, c=dcolors, s=3, lw=0)

    for row, (name, wf) in zip(panels[1:], waveforms.items()):
        row_axs = row.subplots(ncols=2)
        row.suptitle(name, fontsize="small")
        for ax, cs in zip(row_axs, (dcolors, tcolors)):
            geomplot(
                waveforms=wf,
                channels=tchans.numpy(force=True),
                geom=mix_data.prgeom,
                colors=cs,
                subar=True,
                ax=ax,
                max_abs_amp=maa,
                lw=0.5,
            )
            ax.axis("off")

    return fig


def _bold(x):
    return f"$\\mathbf{{{x}}}$"
