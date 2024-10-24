from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import torch
from tqdm.auto import tqdm

from ..cluster import gaussian_mixture
from ..util.multiprocessing_util import (CloudpicklePoolExecutor,
                                         ThreadPoolExecutor, get_pool, cloudpickle)
from . import analysis_plots, gmm_helpers, layout
from .colors import glasbey1024
from .waveforms import geomplot


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
    height = 2

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
        axis.set_ylabel(f"count ({dt_ms.size+1} tot. sp.)")


class ChansHeatmap(GMMPlot):
    kind = "tall"
    width = 2
    height = 3

    def __init__(self, cmap=plt.cm.magma):
        self.cmap = cmap

    def draw(self, panel, gmm, unit_id):
        (in_unit_full,) = torch.nonzero(gmm.labels == unit_id, as_tuple=True)
        spike_chans = gmm.data.extract_channels[in_unit_full].numpy(force=True)
        ixs = spike_chans[spike_chans < gmm.data.n_channels]
        unique_ixs, counts = np.unique(ixs, return_counts=True)
        ax = panel.subplots()
        xy = gmm.data.prgeom.numpy(force=True)
        s = ax.scatter(*xy[unique_ixs].T, c=counts, lw=0, cmap=self.cmap)
        plt.colorbar(s, ax=ax, shrink=0.3)
        ax.scatter(
            *xy[gmm.units[unit_id].channels.numpy(force=True)].T,
            color="r",
            lw=1,
            fc="none",
        )
        ax.scatter(
            *xy[np.atleast_1d(gmm.units[unit_id].snr.argmax().numpy(force=True))].T,
            color="g",
            lw=0,
        )


class TextInfo(GMMPlot):
    kind = "aaatext"
    width = 2
    height = 1

    def draw(self, panel, gmm, unit_id):
        axis = panel.subplots()
        axis.axis("off")
        msg = f"unit {unit_id}\n"

        nspikes = (gmm.labels == unit_id).sum()
        msg += f"n spikes: {nspikes}\n"

        axis.text(0, 0, msg, fontsize=6.5)


class MStep(GMMPlot):
    kind = "waveform"
    width = 4
    height = 5
    alpha = 0.05
    n_show = 64

    def draw(self, panel, gmm, unit_id, axes=None):
        ax = panel.subplots()
        ax.axis("off")

        sp = gmm.random_spike_data(unit_id, max_size=self.n_show, with_reconstructions=True)
        maa = sp.waveforms.abs().nan_to_num().max()
        lines, chans = geomplot(
            sp.waveforms,
            channels=sp.channels,
            geom=gmm.data.prgeom.numpy(force=True),
            max_abs_amp=maa,
            color="k",
            alpha=self.alpha,
            return_chans=True,
            show_zero=False,
            ax=ax,
        )
        chans = torch.tensor(list(chans))
        tup = gaussian_mixture.to_full_probe(
            sp, weights=None, n_channels=gmm.data.n_channels, storage=None
        )
        features_full, weights_full, count_data, weights_normalized = tup
        emp_mean = torch.nanmean(features_full, dim=0)[:, chans]
        emp_mean = gmm.data.tpca.force_reconstruct(emp_mean.nan_to_num_())

        model_mean = gmm.units[unit_id].mean[:, chans]
        model_mean = gmm.data.tpca.force_reconstruct(model_mean)

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


class Likelihoods(GMMPlot):
    kind = "widescatter"
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
        inds_, liks = gmm.unit_log_likelihoods(unit_id, spike_indices=in_unit)
        assert torch.equal(inds_, in_unit)
        nliks = gmm.noise_log_likelihoods()[1][in_unit]
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
        n, bins, _ = ax_dist.hist(liks, color=c, label="unit", bins=64, **histk)
        ax_dist.hist(nliks, color="k", label="noise", bins=bins, **histk)


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
    width = 5
    height = 9

    def draw(self, panel, gmm, unit_id, axes=None):
        split_info = gmm.kmeans_split_unit(unit_id, debug=True)
        split_labels = split_info["merge_labels"]
        split_ids = np.unique(split_labels)

        amps_row, centroids_row, modes_row = panel.subfigures(nrows=3, height_ratios=[1.5, 2, 2])
        fig_chans, fig_dists = modes_row.subfigures(ncols=2)
        fig_dist, fig_bimods = fig_dists.subfigures(nrows=2)
        panel.suptitle('kmeans split info')

        # subunit amplitudes
        gmm_helpers.amp_double_scatter(
            gmm, split_info["sp"].indices, amps_row, labels=split_info["merge_labels"]
        )

        # distance matrix
        ax_dist = analysis_plots.distance_matrix_dendro(
            fig_dist,
            split_info["distances"],
            # unit_ids=split_ids,
            dendrogram_linkage=None,
            show_unit_labels=True,
            vmax=1.0,
            image_cmap=plt.cm.RdGy,
            show_values=True,
        )
        normstr = ", noisenormed" if gmm.distance_noise_normalized else ""
        ax_dist.set_title(f"{gmm.distance_metric}{normstr}", fontsize="small")

        # bimodality matrix
        ax_bimod = analysis_plots.distance_matrix_dendro(
            fig_bimods,
            split_info["bimodalities"],
            # unit_ids=split_ids,
            dendrogram_linkage=None,
            show_unit_labels=True,
            vmax=0.5,
            image_cmap=plt.cm.RdGy,
            show_values=True,
        )
        ax_bimod.set_title("bimodality")

        # subunit means on the unit main channel, where possible
        ax_centroids, ax_mycentroids = centroids_row.subplots(ncols=2, sharey=True)
        ax_centroids.set_title("orig unit main chan", fontsize="small")
        ax_mycentroids.set_title("split unit main chan", fontsize="small")
        for ax in (ax_centroids, ax_mycentroids):
            ax.set_xticks([])
            ax.axhline(0, color="k", lw=0.8)
            sns.despine(ax=ax, left=False, right=True, bottom=True, top=True)
        mainchan = gmm.units[unit_id].snr.argmax()
        for subid, subunit in zip(split_ids, split_info["units"]):
            subm = subunit.mean[:, mainchan]
            subm = gmm.data.tpca._inverse_transform_in_probe(subm[None])[0]
            ax_centroids.plot(subm, color=glasbey1024[subid])

            subm = subunit.mean[:, subunit.snr.argmax()]
            subm = gmm.data.tpca._inverse_transform_in_probe(subm[None])[0]
            ax_mycentroids.plot(subm, color=glasbey1024[subid])

        # subunit channels histogram
        chan_bins = torch.unique(split_info["sp"].channels)
        chan_bins = chan_bins[chan_bins < gmm.data.n_channels]
        chan_bins = torch.arange(chan_bins.min(), chan_bins.max() + 1)
        unit_chans = []
        for j in split_ids:
            uc = split_info["sp"].channels[split_labels == j]
            unit_chans.append(uc[uc < gmm.data.n_channels])
        ax_chans = fig_chans.subplots()
        ax_chans.hist(
            unit_chans,
            histtype="bar",
            bins=chan_bins,
            color=glasbey1024[split_ids],
            stacked=True,
        )


# -- merge-oriented plots


class NeighborMeans(GMMPlot):
    kind = "merge"
    width = 3
    height = 4

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def draw(self, panel, gmm, unit_id):
        neighbors = gmm_helpers.get_neighbors(gmm, unit_id)
        # means on core channels
        chans = gmm.units[unit_id].snr.argmax()
        chans = torch.cdist(gmm.data.prgeom[chans[None]], gmm.data.prgeom)
        chans = chans.view(-1)
        (chans,) = torch.nonzero(chans <= gmm.data.core_radius, as_tuple=True)
        means = []
        for u in reversed(neighbors):
            mean = gmm.units[u].mean[:, chans]
            means.append(gmm.data.tpca.force_reconstruct(mean).numpy(force=True))

        ax = panel.subplots()
        labels = neighbors.numpy(force=True)[::-1]
        colors = glasbey1024[labels]
        geomplot(
            np.stack(means, axis=0),
            channels=chans[None]
            .broadcast_to(len(means), *chans.shape)
            .numpy(force=True),
            geom=gmm.data.prgeom.numpy(force=True),
            colors=colors,
            show_zero=False,
            ax=ax,
        )
        panel.legend(
            handles=[Line2D([0, 1], [0, 0], color=c) for c in colors],
            labels=labels.tolist(),
            loc="outside upper center",
            frameon=False,
            ncols=3,
            title="nearest neighbors",
        )
        ax.axis("off")


class NeighborDistances(GMMPlot):
    kind = "merge"
    width = 3
    height = 3

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def draw(self, panel, gmm, unit_id):
        neighbors = gmm_helpers.get_neighbors(gmm, unit_id)
        distances = gmm.distances(
            units=[gmm.units[u] for u in neighbors], show_progress=False
        )
        ax = analysis_plots.distance_matrix_dendro(
            panel,
            distances,
            unit_ids=neighbors.numpy(force=True),
            dendrogram_linkage=None,
            show_unit_labels=True,
            vmax=0.5,
            image_cmap=plt.cm.RdGy,
            show_values=True,
        )
        normstr = ", noisenormed" if gmm.distance_noise_normalized else ""
        ax.set_title(f"nearby {gmm.distance_metric}{normstr}", fontsize="small")


class NeighborBimodalities(GMMPlot):
    kind = "merge"
    width = 4
    height = 8

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def draw(self, panel, gmm, unit_id):
        neighbors = gmm_helpers.get_neighbors(gmm, unit_id)
        assert neighbors[0] == unit_id
        log_liks = gmm.log_likelihoods(unit_ids=neighbors)
        labels = gaussian_mixture.loglik_reassign(log_liks, has_noise_unit=True)
        log_liks = gaussian_mixture.coo_to_torch(log_liks, torch.float)
        kept = labels >= 0
        labels_ = np.full_like(labels, -1)
        labels_[kept] = neighbors[labels[kept]].numpy(force=True)
        labels = labels_

        others = neighbors[1:]
        axes = panel.subplots(nrows=len(others), ncols=2)
        histkw = dict(density=True, histtype="step", bins=128)
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
            c = glasbey1024[labels[bimod_info["in_pair_kept"]]]
            scatter_ax.scatter(bimod_info["xi"], bimod_info["xj"], s=3, lw=0, c=c)
            scatter_ax.set_ylabel(unit_id, color=glasbey1024[unit_id])
            scatter_ax.set_xlabel(other_id.item(), color=glasbey1024[other_id])

            if "samples" not in bimod_info:
                bimod_ax.text(0, 0, f"kept {bimod_info['keep_prop']:0.3f}")
                bimod_ax.axis("off")
                continue
            bimod_ax.hist(bimod_info["samples"], color="gray", **histkw)
            bimod_ax.hist(
                bimod_info["samples"],
                weights=bimod_info["sample_weights"],
                color="k",
                **histkw,
            )
            bimod_ax.axvline(bimod_info["cut"], color="k", lw=0.8, ls=":")
            bimod_ax.plot(
                bimod_info["domain"], bimod_info["alternative_density"], color="r"
            )
            bimod_ax.plot(bimod_info["domain"], bimod_info["uni_density"], color="b")
            info = f"{bimod_info['score_kind']}{bimod_info['score']:.3f} in-{other_id}-ness"
            bimod_ax.set_xlabel(info)
            bimod_ax.set_yticks([])

# -- main api

default_gmm_plots = (
    TextInfo(),
    ISIHistogram(),
    ChansHeatmap(),
    MStep(),
    Likelihoods(),
    Amplitudes(),
    KMeansSplit(),
    NeighborMeans(),
    NeighborDistances(),
    NeighborBimodalities(),
)


def make_unit_gmm_summary(
    gmm,
    unit_id,
    plots=default_gmm_plots,
    max_height=9,
    figsize=(12, 8),
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
    figsize=(12, 8),
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
        initargs = (cloudpickle.dumps(initargs),)
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
