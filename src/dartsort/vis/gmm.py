import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm.auto import tqdm

from dartsort.util.drift_util import get_shift_info, get_spike_pitch_shifts
from dartsort.util.waveform_util import get_pitch

from ..cluster import gaussian_mixture, stable_features
from ..util import spiketorch
from ..util.multiprocessing_util import (
    CloudpicklePoolExecutor,
    ThreadPoolExecutor,
    cloudpickle,
    get_pool,
)
from . import analysis_plots, gmm_helpers, layout
from .colors import glasbey1024
from .waveforms import geomplot

distance_cmap = plt.cm.plasma


class GMMPlot(layout.BasePlot):
    width = 1
    height = 1
    kind = "gmm"

    def draw(self, panel, gmm, unit_id):
        raise NotImplementedError


# -- summary plots


class ISIHistogram(GMMPlot):
    kind = "small"
    width = 2
    height = 1.5

    def __init__(self, bin_ms=0.1, max_ms=5):
        self.bin_ms = bin_ms
        self.max_ms = max_ms

    def draw(self, panel, gmm, unit_id):
        axis = panel.subplots()
        times_s = gmm.data.times_seconds[gmm.labels == unit_id]
        dt_ms = np.diff(times_s) * 1000
        bin_edges = np.arange(0, self.max_ms + self.bin_ms, self.bin_ms)
        counts, _ = np.histogram(dt_ms, bin_edges)
        axis.stairs(counts, bin_edges, color=glasbey1024[unit_id], fill=True)
        axis.set_xlabel("isi (ms)")
        axis.set_ylabel(f"count ({dt_ms.size + 1} tot. sp.)")


class ChansHeatmap(GMMPlot):
    kind = "tall"
    width = 2
    height = 3

    def __init__(self, cmap=plt.cm.magma):
        self.cmap = cmap

    def draw(self, panel, gmm, unit_id):
        train_ix = gmm.data.split_indices["train"]
        (in_unit_full,) = torch.nonzero(gmm.labels[train_ix] == unit_id, as_tuple=True)
        spike_chans = gmm.data._train_extract_channels[in_unit_full].numpy(force=True)
        ixs = spike_chans[spike_chans < gmm.data.n_channels]
        unique_ixs, counts = np.unique(ixs, return_counts=True)
        ax = panel.subplots()
        xy = gmm.data.prgeom.numpy(force=True)
        s = ax.scatter(*xy[unique_ixs].T, c=counts, lw=0, cmap=self.cmap)
        plt.colorbar(s, ax=ax, shrink=0.3, label="chan count")
        ax.scatter(
            *xy[gmm[unit_id].channels.numpy(force=True)].T,
            color="r",
            lw=1,
            fc="none",
        )
        ax.scatter(
            *xy[np.atleast_1d(gmm[unit_id].snr.argmax().numpy(force=True))].T,
            color="g",
            lw=0,
        )


class TextInfo(GMMPlot):
    kind = "aaatext"
    width = 2
    height = 3

    def __init__(self, title=None):
        self.title = title

    def draw(self, panel, gmm, unit_id):
        axis = panel.subplots()
        axis.axis("off")
        msg = f"unit {unit_id}\n"

        if self.title:
            axis.set_title(f"{self.title} {unit_id}")

        nspikes = (gmm.labels == unit_id).sum()
        msg += f"n spikes: {nspikes}\n"
        if gmm[unit_id].annotations:
            msg += "annots:\n"
            for k, v in gmm[unit_id].annotations.items():
                if torch.is_tensor(k):
                    k = k.numpy(force=True)
                    if k.size == 1:
                        k = k.item()
                if torch.is_tensor(v):
                    v = v.numpy(force=True)
                if isinstance(v, (np.ndarray, list, tuple)):
                    v = np.array2string(
                        np.asarray(v),
                        separator=",",
                        precision=1,
                        threshold=100,
                        max_line_width=32,
                    )
                msg += f"{k}:\n{v}"

        axis.text(0, 0, msg, fontsize=5.5)


class MStep(GMMPlot):
    kind = "mstep"
    width = 5
    height = 4
    alpha = 0.05

    def __init__(self, n_waveforms_show=64):
        self.n_waveforms_show = n_waveforms_show

    def draw(self, panel, gmm, unit_id, axes=None):
        ax = panel.subplots()
        ax.axis("off")

        # get spike data and determine channel set by plotting
        sp = gmm.random_spike_data(
            unit_id, max_size=self.n_waveforms_show, with_reconstructions=True
        )
        maa = sp.waveforms.abs().nan_to_num().max().numpy(force=True)
        geomplot_kw = dict(
            max_abs_amp=maa,
            geom=gmm.data.prgeom.numpy(force=True),
            show_zero=False,
            return_chans=True,
        )
        lines, chans = geomplot(
            sp.waveforms.numpy(force=True),
            channels=sp.channels.numpy(force=True),
            color="k",
            alpha=self.alpha,
            ax=ax,
            **geomplot_kw,
        )
        chans = torch.tensor(list(chans))
        tup = gaussian_mixture.to_full_probe(
            sp, weights=None, n_channels=gmm.data.n_channels, storage=None
        )
        features_full, weights_full, count_data, weights_normalized = tup
        feats = features_full[:, :, chans]
        n, r, c = feats.shape
        emp_mean = torch.nanmean(feats, dim=0)
        emp_mean = gmm.data.tpca.force_reconstruct(emp_mean.nan_to_num_()).numpy(
            force=True
        )
        model_mean = gmm[unit_id].mean[:, chans]
        model_mean = gmm.data.tpca.force_reconstruct(model_mean).numpy(force=True)

        geomplot(
            np.stack([emp_mean, model_mean], axis=0),
            channels=chans[None].broadcast_to(2, *chans.shape).numpy(force=True),
            max_abs_amp=maa,
            geom=gmm.data.prgeom.numpy(force=True),
            colors=["k", glasbey1024[unit_id]],
            show_zero=False,
            ax=ax,
        )
        ax.axis("off")
        ax.set_title("reconstructed mean and example inputs", fontsize="small")


class CovarianceResidual(GMMPlot):
    kind = "mstep"
    width = 7
    height = 5

    def draw(self, panel, gmm, unit_id):
        sp = gmm.random_spike_data(unit_id)
        weights = gmm.get_fit_weights(
            unit_id, sp.indices, getattr(gmm, "log_liks", None)
        )

        # achans = gaussian_mixture.occupied_chans(sp, gmm.noise.n_channels)
        achans = gmm[unit_id].channels
        if weights is None:
            weights = sp.features.new_ones(len(sp))
        afeats, aweights = stable_features.pad_to_chans(
            sp, achans, gmm.noise.n_channels, weights=weights, pad_value=torch.nan
        )
        aweights_sum = torch.nansum(aweights, 0)
        aweights_norm = aweights / aweights_sum

        mean = torch.linalg.vecdot(
            aweights_norm.unsqueeze(1).nan_to_num(), afeats.nan_to_num(), dim=0
        )
        afeatsc = afeats - mean

        emp_cov = spiketorch.nancov(
            afeatsc.view(len(sp), -1), weights=weights, correction=0, force_posdef=True
        )
        # emp_cov = torch.cov(afeatsc.view(len(sp), -1).nan_to_num().T)
        noise_cov = gmm.noise.marginal_covariance(achans).to_dense()
        residual = emp_cov - noise_cov

        mmT = mean.view(-1, 1) @ mean.view(1, -1)
        scale = (mmT * residual).sum() / mmT.square().sum()
        if scale < 0:
            scale = 0.0
        mmT_cov = noise_cov + scale * mmT
        mmT_residual = emp_cov - mmT_cov

        emp_eigs = torch.linalg.eigvalsh(emp_cov)
        noise_eigs = torch.linalg.eigvalsh(noise_cov)
        residual_eigs, residual_vecs = torch.linalg.eigh(residual)
        if scale == 0:
            mmT_residual_eigs = torch.zeros(mmT_residual.shape[0])
        else:
            try:
                mmT_residual_eigs = torch.linalg.eigvalsh(mmT_residual)
            except Exception as e:
                warnings.warn(f"mmT residual. {e}")
                mmT_residual_eigs = torch.zeros(mmT_residual.shape[0])

        rank1 = (residual_vecs[:, -1:] * residual_eigs[-1:]) @ residual_vecs[:, -1:].T
        rank1_model = noise_cov + rank1
        rank1_residual = emp_cov - rank1_model
        rank1_residual_eigs = torch.linalg.eigvalsh(rank1_residual)

        signal = gmm[unit_id].marginal_covariance(channels=achans, signal_only=True)
        signal = signal.to_dense()
        modelcov = gmm[unit_id].marginal_covariance(channels=achans)
        modelcov = modelcov.to_dense()
        model_residual = emp_cov - modelcov
        model_residual_eigs = torch.linalg.eigvalsh(model_residual)

        top, bot = panel.subfigures(nrows=2, height_ratios=[5, 2])
        axes = top.subplots(nrows=4, ncols=3, sharex=True, sharey=True)
        # ax_eig, ax_r2 = bot.subplots(ncols=2)
        ax_eig = bot.subplots()

        # vm = 0.9 * emp_cov.abs().max()
        imk = dict(cmap=plt.cm.seismic, interpolation="none")

        covs = dict(
            emp=emp_cov,
            noise=noise_cov,
            noise_resid=residual,
            mmT=scale * mmT,
            mmT_model=mmT_cov,
            mmT_resid=mmT_residual,
            rank1=rank1,
            rank1_model=rank1_model,
            rank1_resid=rank1_residual,
            model_signal=signal,
            model=modelcov,
            model_resid=model_residual,
        )
        colors = dict(
            emp="k",
            noise="g",
            noise_resid="r",
            mmT="gray",
            mmT_model="gray",
            mmT_resid="orange",
            rank1="gray",
            rank1_model="gray",
            rank1_resid="fuchsia",
            model_signal="gray",
            model="gray",
            model_resid="purple",
        )

        eigs = dict(
            emp=emp_eigs,
            noise=noise_eigs,
            noise_resid=residual_eigs,
            mmT_resid=mmT_residual_eigs,
            rank1_resid=rank1_residual_eigs,
            model_resid=model_residual_eigs,
        )
        for (name, cov), ax in zip(covs.items(), axes.flat):
            color = colors[name]
            vm = cov.abs().max().numpy(force=True) * 0.9
            mimk = imk | dict(vmax=vm, vmin=-vm)
            im = ax.imshow(cov.numpy(force=True), **mimk)
            cb = plt.colorbar(im, ax=ax, shrink=0.2)
            cb.outline.set_visible(False)
            title = name
            if name == "mmT":
                title = title + f" (scale={scale:.2f})"
            ax.set_title(title, color=color, fontsize="small")
            if name in eigs:
                ax_eig.plot(eigs[name].flip(0).numpy(force=True), color=color, lw=1)
                # r2 = (eigs['emp'].sum() - F.relu(eigs[name].flip(0)).cumsum(0)) / eigs['emp'].sum()
                # ax_r2.plot(r2, color=color, lw=1)

        for ax in axes.flat[len(covs) :]:
            ax.axis("off")
        ax_eig.set_xlim([-0.05, 25.05])
        ax_eig.set_xlabel("eig index")
        ax_eig.set_ylabel("eigenvalues")
        ax_eig.axhline(0, color="k", lw=0.8)
        # ax_r2.set_ylabel('1-R^2')
        # ax_eig.axhline([0, 1], color='k', lw=0.8)


class WaveformCheck(GMMPlot):
    kind = "mstep"
    width = 5
    height = 4.5

    def __init__(
        self,
        neighborhood="extract",
        split="train",
        colorvar="displacement",
        cmap="viridis",
        localizations_name="localizations",
        randomize=True,
    ):
        assert colorvar in (
            "time",
            "depth",
            "chandepth",
            "displacement",
            "npitches",
            "subpitch",
        )
        self.neighborhood = neighborhood
        self.split = split
        self.colorvar = colorvar
        self.cmap = plt.get_cmap(cmap)
        self.localizations_name = localizations_name
        self.randomize = randomize

    def draw(self, panel, gmm, unit_id, axes=None):
        s = object()
        me = getattr(gmm, "motion_est", s)
        if me is s:
            raise ValueError(
                f"Sorry, hacky, but to use {self.__class__.__name__} you need to "
                "assign the motion estimate as the property .motion_est of the GMM."
            )

        _, ixs, splitixs = gmm.random_indices(unit_id=unit_id, split_name=self.split)
        sp = gmm.data.spike_data(
            ixs,
            split_indices=splitixs,
            with_reconstructions=True,
            neighborhood=self.neighborhood,
        )

        if self.colorvar == "time":
            c = gmm.data.times_seconds[ixs].numpy(force=True)
        elif self.colorvar == "chandepth":
            chans = gmm.data.original_sorting.channels[ixs]
            c = gmm.data.original_sorting.geom[chans, 1]
        elif self.colorvar == "depth":
            pos = getattr(self.data.original_sorting, self.localizations_name)
            c = pos[ixs, 2]
        else:
            channels, shifts, n_pitches_shift = get_shift_info(
                gmm.data.original_sorting,
                motion_est=me,
                geom=gmm.data.original_sorting.geom,
            )
            shifts = shifts[ixs]
            n_pitches_shift = n_pitches_shift[ixs]
            if self.colorvar == "displacement":
                c = shifts
            elif self.colorvar == "npitches":
                c = n_pitches_shift
            elif self.colorvar == "subpitch":
                pitch = get_pitch(gmm.data.original_sorting.geom)
                c = shifts - pitch * n_pitches_shift
            else:
                assert False

        ax = panel.subplots()
        s = geomplot(
            sp.waveforms.numpy(),
            channels=sp.channels.numpy(),
            geom=gmm.data.prgeom[:-1].numpy(),
            c=self.cmap(minmax(c)),
            alpha=0.1,
            ax=ax,
            randomize=self.randomize,
        )
        ax.axis("off")
        st = f"{self.split}/{self.neighborhood} by {self.colorvar}"
        if self.randomize:
            st += ", rand order"
        ax.set_title(st)


class Likelihoods(GMMPlot):
    kind = "merge"
    width = 4
    height = 2

    def __init__(self, viol_ms=1.0):
        self.viol_ms = viol_ms

    def draw(self, panel, gmm, unit_id, axes=None):
        ax_time, ax_noise, ax_dist = panel.subplots(
            ncols=3, width_ratios=[3, 2, 1], sharey=True
        )
        ax_time.set_ylabel("log likelihood")
        ax_time.set_xlabel("time (s)")
        ax_noise.set_xlabel("noise loglik")
        ax_dist.set_xlabel("count")
        (in_unit,) = torch.nonzero(gmm.labels == unit_id, as_tuple=True)
        if not in_unit.numel():
            return
        if hasattr(gmm, "log_liks"):
            liks_ = gmm.log_liks[:, in_unit.numpy(force=True)][[unit_id]].tocoo()
            inds_ = None
            if liks_.nnz:
                inds_ = in_unit
                liks = np.full(in_unit.shape, -np.inf, dtype=np.float32)
                liks[liks_.coords[1]] = liks_.data
                liks = torch.from_numpy(liks)
        else:
            inds_, liks = gmm.unit_log_likelihoods(unit_id, spike_indices=in_unit)
        if inds_ is None:
            return
        assert torch.equal(inds_, in_unit)
        nliks = gmm.noise_log_likelihoods()[in_unit]
        t = gmm.data.times_seconds[in_unit]
        dt_ms = np.diff(t) * 1000
        small = dt_ms <= self.viol_ms

        c = glasbey1024[unit_id]
        ax_time.scatter(t, liks, s=3, lw=0, color=c)
        if small.any():
            small = np.logical_or(
                np.pad(small, (1, 0), constant_values=False),
                np.pad(small, (0, 1), constant_values=False),
            )
            ax_time.scatter(t[small], liks[small], s=3, lw=0, color="k")
        ax_noise.scatter(nliks, liks, s=3, lw=0, color=c)
        histk = dict(histtype="step", orientation="horizontal")
        n, bins, _ = ax_dist.hist(
            liks[torch.isfinite(liks)], color=c, label="unit", bins=64, **histk
        )
        ax_dist.hist(nliks, color="k", label="noise", bins=bins, **histk)
        ax_dist.legend(
            loc="lower right",
            borderpad=0.1,
            frameon=False,
            borderaxespad=0.1,
            handletextpad=0.3,
            handlelength=1.0,
        )


class Amplitudes(GMMPlot):
    kind = "widescatter"
    width = 4
    height = 2

    def __init__(self, viol_ms=1.0):
        self.viol_ms = viol_ms

    def draw(self, panel, gmm, unit_id, axes=None):
        (in_unit,) = torch.nonzero(gmm.labels == unit_id, as_tuple=True)
        gmm_helpers.amp_double_scatter(
            gmm, in_unit, panel, unit_id=unit_id, labels=None, viol_ms=self.viol_ms
        )


# -- split-oriented plots


class KMeansSplit(GMMPlot):
    kind = "friend"
    width = 6.5
    height = 9

    def __init__(
        self,
        criterion=None,
        layout="vert",
        neighborhood="core",
        with_means=True,
        decision_algorithm=None,
        ignore_channels=None,
        kmeans_n_iter=None,
        min_overlap=None,
        metric=None,
        distance_normalization_kind=None,
    ):
        self.layout = layout
        self.neighborhood = neighborhood
        if criterion and criterion.startswith("brute_"):
            decision_algorithm = "brute"
            criterion = criterion.removeprefix("brute_")
        self.criterion = criterion
        self.with_means = with_means
        self.decision_algorithm = decision_algorithm
        self.ignore_channels = ignore_channels
        self.kmeans_n_iter = kmeans_n_iter
        self.min_overlap = min_overlap
        self.metric = metric
        self.distance_normalization_kind = distance_normalization_kind

    def draw(self, panel, gmm, unit_id, split_info=None):
        criterion = self.criterion or gmm.criterion
        ickw = {}
        if self.ignore_channels is not None:
            ickw["ignore_channels"] = self.ignore_channels
        min_overlap = self.min_overlap
        if min_overlap is None:
            min_overlap = gmm.min_overlap
        metric = self.metric or gmm.distance_metric
        normkind = self.distance_normalization_kind or gmm.distance_normalization_kind

        if split_info is None:
            split_info = gmm.kmeans_split_unit(
                unit_id,
                debug=True,
                criterion=criterion,
                decision_algorithm=self.decision_algorithm,
                kmeans_n_iter=self.kmeans_n_iter,
                min_overlap=min_overlap,
                distance_metric=metric,
                distance_normalization_kind=normkind,
                **ickw,
            )
        failed0 = not split_info
        failed1 = "reas_labels" not in split_info
        failed = failed0 or failed1
        if failed:
            ax = panel.subplots()
            if not split_info:
                msg = "no channels!"
            else:
                msg = "split abandoned" + split_info.get("bail", "no bail")
            ax.text(0.5, 0.5, msg, ha="center", transform=ax.transAxes)
            ax.axis("off")
            return

        split_labels = split_info["split_labels"]
        split_ids = np.unique(split_labels)
        if self.with_means:
            kw = dict(nrows=4, height_ratios=[1, 1, 2, 2])
            if self.layout == "horz":
                kw = dict(ncols=4, width_ratios=[1, 1, 1, 1])
            amps_row, centroids_row, mcmeans_row, modes_row = panel.subfigures(**kw)
        else:
            kw = dict(nrows=2, height_ratios=[1, 2])
            if self.layout == "horz":
                kw = dict(ncols=2)
            amps_row, modes_row = panel.subfigures(**kw)

        fig_chans, fig_dists = modes_row.subfigures(ncols=2)
        fig_dist, fig_bimods = fig_dists.subfigures(nrows=2)
        panel.suptitle("kmeans split info")

        # subunit amplitudes
        gmm_helpers.amp_double_scatter(
            gmm, split_info["sp"].indices, amps_row, labels=split_labels
        )

        # distance matrix
        if "distances" in split_info:
            ax_dist = analysis_plots.distance_matrix_dendro(
                fig_dist,
                split_info["distances"],
                # unit_ids=split_ids,
                dendrogram_linkage=None,
                show_unit_labels=True,
                vmax=1.0,
                image_cmap=distance_cmap,
                show_values=True,
            )
            normstr = f"norm={normkind}"
            ax_dist.set_title(f"{metric}, {normstr}", fontsize="small")

        if "bimodalities" in split_info:
            # bimodality matrix
            ax_bimod = analysis_plots.distance_matrix_dendro(
                fig_bimods,
                split_info["bimodalities"],
                # unit_ids=split_ids,
                dendrogram_linkage=None,
                show_unit_labels=True,
                vmax=0.5,
                image_cmap=distance_cmap,
                show_values=True,
            )
            ax_bimod.set_title("bimodality", fontsize="small")
        elif "Z" in split_info:
            assert "improvements" in split_info
            ax_bimod = fig_bimods.subplots()
            improvements = split_info["improvements"]
            olaps = np.floor(split_info["overlaps"] * 100)
            annotations = {
                j: f"{imp:.2f} {olaps[j]:g}" for j, imp in enumerate(improvements)
            }
            analysis_plots.annotated_dendro(
                ax_bimod,
                split_info["Z"],
                annotations,
                threshold=gmm.merge_distance_threshold,
                annotations_offset_by_n=False,
            )
            ax_bimod.set_title(
                f"tree {gmm.distance_metric} {criterion}-{gmm.criterion_normalization_kind}",
                fontsize="small",
            )
            sns.despine(ax=ax_bimod, left=True, right=True, top=True)
            if "full_improvement" in split_info:
                ax_bimod.set_xlabel(f"full split: {split_info['full_improvement']:.3f}")
        elif "ids_part" in split_info:
            ax_bimod = fig_bimods.subplots()
            ax_bimod.axis("off")
            imp = split_info["improvements"][0]
            ax_bimod.text(
                0.5,
                0.5,
                f"{self.decision_algorithm} | {criterion}\n"
                f"{split_info['ids_part']}\n"
                f"imp:{imp:0.3f}\n"
                f"olap:{split_info['overlap']:0.3f}\n"
                f"full imp: {split_info['full_improvement']:0.3f}\n",
                ha="center",
                va="center",
                fontsize=6,
                transform=ax_bimod.transAxes,
            )
        else:
            ax_bimod = fig_bimods.subplots()
            ax_bimod.text(
                0.5, 0.5, "no mini merge", ha="center", transform=ax_bimod.transAxes
            )
            ax_bimod.axis("off")

        # subunit means on the unit main channel, where possible
        if self.with_means:
            if layout == "vert":
                ax_centroids, ax_mycentroids = centroids_row.subplots(
                    ncols=2, sharey=True
                )
            else:
                ax_centroids, ax_mycentroids = centroids_row.subplots(
                    nrows=2, sharex=True, sharey=True
                )
            ax_centroids.set_ylabel("orig unit main chan")
            ax_mycentroids.set_ylabel("split unit main chan")
            for ax in (ax_centroids, ax_mycentroids):
                ax.set_xticks([])
                ax.axhline(0, color="k", lw=0.8)
                sns.despine(ax=ax, left=False, right=True, bottom=True, top=True)
            mainchan = gmm[unit_id].snr.argmax()
            for subid, subunit in zip(split_ids, split_info["units"]):
                subm = subunit.mean[:, mainchan]
                subm = gmm.data.tpca._inverse_transform_in_probe(subm[None])[0]
                ax_centroids.plot(subm.numpy(force=True), color=glasbey1024[subid])

                subm = subunit.mean[:, subunit.snr.argmax()]
                subm = gmm.data.tpca._inverse_transform_in_probe(subm[None])[0]
                ax_mycentroids.plot(subm.numpy(force=True), color=glasbey1024[subid])

            subm = gmm[unit_id].mean[:, mainchan]
            subm = gmm.data.tpca._inverse_transform_in_probe(subm[None])[0]
            ax_centroids.plot(subm.numpy(force=True), color="k", lw=0.5)
            ax_mycentroids.plot(subm.numpy(force=True), color="k", lw=0.5)

            # subunit multichan means
            if self.neighborhood == "core":
                chans = torch.cdist(gmm.data.prgeom[mainchan[None]], gmm.data.prgeom)
                chans = chans.view(-1)
                (chans,) = torch.nonzero(chans <= gmm.data.core_radius, as_tuple=True)
            elif self.neighborhood == "unit":
                chans = gmm[unit_id].channels
            else:
                assert False
            if len(split_ids) < len(split_info["units"]):
                split_info["units"] = [split_info["units"][j] for j in split_ids]
            if "units" in split_info:
                try:
                    gmm_helpers.plot_means(
                        mcmeans_row,
                        gmm.data.prgeom,
                        gmm.data.tpca,
                        chans,
                        split_info["units"] + [gmm[unit_id]],
                        list(split_ids) + [-1],
                        title=None,
                        linewidths=[1] * len(split_ids) + [0.5],
                    )
                except Exception:
                    pass

        # subunit channels histogram
        chan_bins = torch.unique(split_info["sp"].channels)
        chan_bins = chan_bins[chan_bins < gmm.data.n_channels]
        chan_bins = torch.arange(chan_bins.min(), chan_bins.max() + 2)
        unit_chans = []
        for j in split_ids:
            uc = split_info["sp"].channels[split_labels == j].cpu()
            unit_chans.append(uc[uc < gmm.data.n_channels])
        ax_pca, ax_chans = fig_chans.subplots(nrows=2)
        ax_chans.hist(
            unit_chans,
            histtype="step",
            bins=chan_bins,
            color=glasbey1024[split_ids],
            log=True,
            # stacked=True,
        )
        ax_chans.set_xlabel("channel")
        ax_chans.set_ylabel("chan count in subunit")

        ax_pca.axis("off")
        if "X" in split_info:
            show_whiten = False and gmm.split_whiten
            key = "X" + "w" * show_whiten

            u, s, v = torch.pca_lowrank(split_info[key].view(len(split_labels), -1))
            Xp = u[:, :2] * s[:2]
            tv = v[:, :2]
            center = split_info["X"].mean(0)

            ax_pca.scatter(*Xp.T, c=glasbey1024[split_labels], s=2, lw=0)
            ax_pca.axhline(0, lw=0.8, color="k")
            ax_pca.axvline(0, lw=0.8, color="k")
            if "units" in split_info:
                for j, u in enumerate(split_info["units"]):
                    gmm_helpers.unit_pca_ellipse(
                        ax=ax_pca,
                        center=center,
                        v=tv,
                        noise=gmm.noise,
                        channels=gmm[unit_id].channels,
                        unit=u,
                        color=glasbey1024[j],
                        whiten=show_whiten,
                        lw=2,
                    )
            if "level_units" in split_info:
                for level, units in reversed(split_info["level_units"].items()):
                    for u in units:
                        gmm_helpers.unit_pca_ellipse(
                            ax=ax_pca,
                            center=center,
                            v=tv,
                            noise=gmm.noise,
                            channels=gmm[unit_id].channels,
                            unit=u,
                            color=glasbey1024[len(split_ids) + level],
                            whiten=show_whiten,
                        )


# -- merge-oriented plots


class NeighborMeans(GMMPlot):
    kind = "merge"
    width = 4
    height = 3

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def draw(self, panel, gmm, unit_id):
        neighbors = gmm_helpers.get_neighbors(gmm, unit_id)
        units = [gmm[u] for u in reversed(neighbors)]
        labels = neighbors[::-1]

        # means on core channels
        chans = gmm[unit_id].snr.argmax()
        chans = torch.cdist(gmm.data.prgeom[chans[None]], gmm.data.prgeom)
        chans = chans.view(-1)
        (chans,) = torch.nonzero(chans <= gmm.data.core_radius, as_tuple=True)

        gmm_helpers.plot_means(
            panel, gmm.data.prgeom, gmm.data.tpca, chans, units, labels
        )


class NeighborDistances(GMMPlot):
    kind = "merge"
    width = 4
    height = 2

    def __init__(
        self, n_neighbors=5, dist_vmax=1.0, metric=None, normalization_kind=None
    ):
        self.n_neighbors = n_neighbors
        self.dist_vmax = dist_vmax
        self.metric = metric
        self.normalization_kind = normalization_kind

    def draw(self, panel, gmm, unit_id):
        neighbors = gmm_helpers.get_neighbors(gmm, unit_id)
        metric = self.metric
        if metric is None:
            metric = gmm.distance_metric
        normalization_kind = self.normalization_kind
        if normalization_kind is None:
            normalization_kind = gmm.distance_normalization_kind
        ids, distances = gmm.distances(
            units=[gmm[u] for u in neighbors],
            show_progress=False,
            kind=metric,
            normalization_kind=normalization_kind,
        )
        ax = analysis_plots.distance_matrix_dendro(
            panel,
            distances,
            unit_ids=neighbors,
            dendrogram_linkage=None,
            show_unit_labels=True,
            vmax=self.dist_vmax,
            image_cmap=distance_cmap,
            show_values=True,
        )
        normstr = f"norm={normalization_kind}"
        ax.set_title(f"nearby {metric}{normstr}", fontsize="small")


class NeighborBimodalities(GMMPlot):
    kind = "bim"
    width = 4
    height = 9

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def draw(self, panel, gmm, unit_id):
        neighbors = gmm_helpers.get_neighbors(gmm, unit_id)
        assert neighbors[0] == unit_id
        if hasattr(gmm, "log_liks"):
            neighbors_plus_noiseunit = np.concatenate(
                (neighbors, [gmm.log_liks.shape[0] - 1])
            )
            log_liks = gmm.log_liks[neighbors_plus_noiseunit]
        else:
            log_liks = gmm.log_likelihoods(unit_ids=neighbors)
        nz_lines, labels_, spikells, log_liks = gaussian_mixture.loglik_reassign(
            log_liks, has_noise_unit=True
        )
        kept = np.flatnonzero(np.logical_and(labels_ >= 0, labels_ < len(neighbors)))
        labels = np.full(log_liks.shape[1], -1)
        labels[nz_lines[kept]] = neighbors[labels_[kept]]

        others = neighbors[1:]
        axes = panel.subplots(nrows=len(others), ncols=2)
        histkw = dict(density=True, histtype="step", bins=128)
        no_leg_yet = True
        for j, (axes_row, other_id) in enumerate(zip(axes, others)):
            bimod_info = gmm.unit_pair_bimodality(
                unit_id,
                other_id,
                log_liks,
                loglik_ix_a=0,
                loglik_ix_b=j + 1,
                debug=True,
            )

            scatter_ax, bimod_ax = axes_row
            # if j == 0:
            #     scatter_ax.set_title("loglik pair scatter", fontsize="small")
            #     bimod_ax.set_title("bimodality computation", fontsize="small")

            if "in_pair_kept" not in bimod_info:
                scatter_ax.text(
                    0.5,
                    0.5,
                    f"too few spikes",
                    transform=scatter_ax.transAxes,
                    ha="center",
                    va="center",
                )
                continue
            else:
                c = np.atleast_2d(glasbey1024[labels[bimod_info["in_pair_kept"]]])
                scatter_ax.scatter(bimod_info["xi"], bimod_info["xj"], s=3, lw=0, c=c)
                scatter_ax.set_ylabel(unit_id, color=glasbey1024[unit_id])
                scatter_ax.set_xlabel(other_id.item(), color=glasbey1024[other_id])

            if "samples" not in bimod_info:
                bimod_ax.text(
                    0.5,
                    0.5,
                    f"too-small\nkept prop {bimod_info['keep_prop']:.2f}",
                    transform=bimod_ax.transAxes,
                    ha="center",
                    va="center",
                )
                bimod_ax.axis("off")
                continue

            bimod_ax.hist(bimod_info["samples"], color="gray", label="hist", **histkw)
            bimod_ax.hist(
                bimod_info["samples"],
                weights=bimod_info["sample_weights"],
                color="k",
                label="whist",
                **histkw,
            )
            bimod_ax.axvline(bimod_info["cut"], color="k", lw=0.8, ls=":")
            bimod_ax.plot(
                bimod_info["domain"],
                bimod_info["alternative_density"],
                color="r",
                label="alt",
            )
            bimod_ax.plot(
                bimod_info["domain"], bimod_info["uni_density"], color="b", label="null"
            )
            info = f"{bimod_info['score_kind']}{bimod_info['score']:.3f} {unit_id}v{other_id.item()}"
            bimod_ax.set_xlabel(info)
            bimod_ax.set_yticks([])
            if no_leg_yet:
                bimod_ax.legend(
                    loc="upper right",
                    borderpad=0.1,
                    frameon=False,
                    borderaxespad=0.1,
                    handletextpad=0.3,
                    handlelength=1.0,
                    markerfirst=False,
                )
                no_leg_yet = False


class NeighborInfoCriteria(GMMPlot):
    kind = "bim"
    width = 4
    height = 9

    def __init__(
        self,
        n_neighbors=5,
        in_bag=False,
    ):
        self.n_neighbors = n_neighbors
        self.in_bag = in_bag

    def draw(self, panel, gmm, unit_id):
        neighbors = gmm_helpers.get_neighbors(gmm, unit_id)
        assert neighbors[0] == unit_id
        others = neighbors[1:]
        axes = panel.subplots(nrows=len(others), ncols=1)
        bag = "inbag" if self.in_bag else "heldout"
        bbox = dict(facecolor="w", alpha=0.5, edgecolor="none")
        histkw = dict(density=True, histtype="step", log=True)
        textkw = dict(bbox=bbox, va="top", fontsize="x-small")

        for ax, other_id in zip(axes, others):
            res = gmm.merge_criteria(
                [unit_id, other_id],
                likelihoods=gmm.log_liks,
                in_bag=self.in_bag,
                debug=True,
            )
            if res is None:
                ax.text(0.5, 0.5, "abandoned", transform=ax.transAxes)
                ax.axis("off")
                continue

            info = res["info"]
            kept = info["keep_mask"].sum() / len(info["keep_mask"])
            for uid, ll in zip((unit_id, other_id), info["subunit_logliks"]):
                if not ll.numel():
                    continue
                ax.hist(ll.cpu(), color=glasbey1024[uid], **histkw)
            merged_ll = info["unit_logliks"]
            if merged_ll.numel():
                ax.hist(merged_ll.cpu(), color="k", **histkw)

            message = f"{100 * kept:.1f}%"
            if "improvements" in res:
                message = f"{message} {bag} full/merge/imp:"
                fc, mc = res["full_criteria"], res["merged_criteria"]
                for k, v in res["improvements"].items():
                    t = ":) " if v >= 0 else "X( "
                    message += f"\n{t}{k}: {fc[k]:.1f} / {mc[k]:.1f} / {v:.2f}"

            ax.text(0.05, 0.95, message, transform=ax.transAxes, **textkw)
            ax.set_xlabel("log lik")
            sns.despine(ax=ax, left=True, right=True, top=True)


class NeighborTreeMerge(GMMPlot):
    kind = "treemerge"
    width = 4
    height = 1.5

    def __init__(
        self,
        n_neighbors=5,
        metric=None,
        criterion=None,
        max_distance=1e10,
        criterion_normalization_kind=None,
        distance_normalization_kind=None,
        threshold=-np.inf,
        decision_algorithm=None,
        min_overlap=None,
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        if criterion and criterion.startswith("brute_"):
            decision_algorithm = "brute"
            criterion = criterion.removeprefix("brute_")
        self.criterion = criterion
        self.max_distance = max_distance
        self.criterion_normalization_kind = criterion_normalization_kind
        self.distance_normalization_kind = distance_normalization_kind
        self.threshold = threshold
        self.decision_algorithm = decision_algorithm
        self.min_overlap = min_overlap

    def draw(self, panel, gmm, unit_id):
        neighbors = gmm_helpers.get_neighbors(gmm, unit_id)
        assert neighbors[0] == unit_id

        criterion = self.criterion
        if criterion is None:
            criterion = gmm.criterion

        metric = self.metric
        if metric is None:
            metric = gmm.distance_metric

        min_overlap = self.min_overlap
        if min_overlap is None:
            min_overlap = gmm.min_overlap

        decision_algorithm = self.decision_algorithm
        if decision_algorithm is None:
            decision_algorithm = gmm.merge_decision_algorithm

        distance_normalization_kind = self.distance_normalization_kind
        if distance_normalization_kind is None:
            distance_normalization_kind = gmm.distance_normalization_kind

        criterion_normalization_kind = self.criterion_normalization_kind
        if criterion_normalization_kind is None:
            criterion_normalization_kind = gmm.criterion_normalization_kind

        ids, distances = gmm.distances(
            units=[gmm[u] for u in neighbors],
            show_progress=False,
            kind=metric,
            normalization_kind=distance_normalization_kind,
        )
        _, cosines = gmm.distances(show_progress=False, kind="cosine")

        if criterion.startswith("old"):
            Z, group_ids, improvements, overlaps = gmm.old_tree_merge(
                distances,
                neighbors,
                max_distance=self.max_distance,
                likelihoods=gmm.log_liks,
                criterion=criterion,
                threshold=self.threshold,
                min_overlap=self.min_overlap,
            )
            brute_indicator = None
        else:
            Z, group_ids, improvements, overlaps, brute_indicator = gmm.tree_merge(
                distances,
                unit_ids=neighbors,
                current_log_liks=gmm.log_liks,
                max_distance=self.max_distance,
                criterion=criterion,
                threshold=self.threshold,
                decision_algorithm=decision_algorithm,
                cosines=cosines,
                min_overlap=self.min_overlap,
            )

        # make vis
        ax = panel.subplots()
        if Z is not None:
            olaps = 0.0 if overlaps is None else np.floor(overlaps * 100)
            annotations = None
            if improvements is not None:
                annotations = {
                    j: f"{imp:.2f} {olaps[j]:g}" for j, imp in enumerate(improvements)
                }
            try:
                analysis_plots.annotated_dendro(
                    ax,
                    Z,
                    annotations,
                    group_ids=group_ids,
                    brute_indicator=brute_indicator,
                    threshold=self.max_distance,
                    leaf_labels=neighbors,
                    annotations_offset_by_n=False,
                )
                nstr = ""
                if distance_normalization_kind != "none":
                    nstr += f"dnm={distance_normalization_kind}"
                if criterion_normalization_kind != "none":
                    nstr += f"cnm={criterion_normalization_kind}"
                ax.set_title(
                    f"{decision_algorithm} {metric} {criterion} {nstr} mo={min_overlap}",
                    fontsize="small",
                )
                sns.despine(ax=ax, left=True, right=True, top=True)
            except ValueError as e:
                ax.text(
                    0.5,
                    0.5,
                    str(e),
                    fontsize="small",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )


# -- main api

default_gmm_plots = (
    TextInfo(),
    ISIHistogram(),
    ISIHistogram(bin_ms=1, max_ms=50),
    ChansHeatmap(),
    MStep(),
    CovarianceResidual(),
    Likelihoods(),
    Amplitudes(),
    KMeansSplit(),
    NeighborMeans(),
    NeighborDistances(metric="noise_metric"),
    NeighborDistances(metric="symkl"),
    NeighborTreeMerge(metric=None, criterion=None),
)


def criterion_comparison_plots(*criteria):
    splits = [KMeansSplit(criterion=k) for k in criteria]
    merges = [NeighborTreeMerge(criterion=k) for k in criteria]
    return (
        TextInfo(),
        ISIHistogram(),
        ISIHistogram(bin_ms=1, max_ms=50),
        ChansHeatmap(),
        MStep(),
        CovarianceResidual(),
        Likelihoods(),
        Amplitudes(),
        *splits,
        NeighborMeans(),
        NeighborDistances(metric="noise_metric"),
        NeighborDistances(metric="symkl"),
        *merges,
    )


figsize = (17, 10)


def make_unit_gmm_summary(
    gmm,
    unit_id,
    plots=default_gmm_plots,
    max_height=9,
    figsize=figsize,
    hspace=0.1,
    figure=None,
    **other_global_params,
):
    # notify plots of global params
    for p in plots:
        p.notify_global_params(
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
    max_height=9,
    figsize=figsize,
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
        unit_ids = gmm.unit_ids()
    if n_units is not None and n_units < len(unit_ids):
        rg = np.random.default_rng(seed)
        unit_ids = rg.choice(unit_ids, size=n_units, replace=False)
    if not overwrite and all_summaries_done(unit_ids, save_folder, ext=image_ext):
        return

    assert hasattr(gmm, "log_liks")

    save_folder.mkdir(exist_ok=True)

    global_params = dict(
        **other_global_params,
    )

    ispar = n_jobs > 0
    cls = CloudpicklePoolExecutor
    if use_threads:
        cls = ThreadPoolExecutor
    n_jobs, Executor, context = get_pool(n_jobs, cls=cls)

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
        rc,
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
        # plt.rcParams = rc


_summary_job_context = None


def _summary_init(*args):
    global _summary_job_context
    # if len(args) == 1:
    #     args = cloudpickle.loads(args[0])
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


def minmax(x):
    y = x - x.min().astype(float)
    y /= np.ptp(y)
    return y
