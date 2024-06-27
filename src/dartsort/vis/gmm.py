from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
from tqdm.auto import tqdm

from ..cluster import density
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

    def __init__(self, kind="residual"):
        self.kind = kind

    def draw(self, panel, gmm, unit_id):
        if self.kind == "residual":
            _, in_unit, z = gmm.split_features(unit_id)
        elif self.kind == "global":
            in_unit, data = gmm.get_training_data(unit_id)
            waveforms = gmm[unit_id].to_unit_channels(
                waveforms=data["waveforms"],
                times=data["times"],
                waveform_channels=data["waveform_channels"],
            )
            loadings, mean, components, svs = spike_interp.fit_pcas(
                data["waveforms"].reshape(in_unit.numel(), -1),
                missing=None,
                empty=None,
                rank=gmm.dpc_split_kw.rank,
                show_progress=False,
            )
            z = loadings.numpy(force=True)

        in_unit = in_unit.numpy(force=True)
        dens = density.density_peaks_clustering(
            z,
            sigma_local=gmm.dpc_split_kw.sigma_local,
            n_neighbors_search=gmm.dpc_split_kw.n_neighbors_search,
            remove_clusters_smaller_than=gmm.min_cluster_size,
            return_extra=True,
        )
        if "density" not in dens:
            ax = panel.subplots()
            ax.text(
                0.5,
                0.5,
                "Clustering threw everyone away",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        ru = np.unique(dens["labels"])
        panel, axes = analysis_plots.density_peaks_study(
            z,
            dens,
            s=10,
            fig=panel,
        )
        axes[-1].set_title(f"n={(ru>=0).sum()}", fontsize=8)
        axes[0].set_title(self.kind)


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


class AmplitudesOverTimePlot(GMMPlot):
    kind = "embeds"
    width = 5
    height = 3

    def draw(self, panel, gmm, unit_id):
        in_unit, utd = gmm.get_training_data(unit_id)
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
        ax.legend(loc="upper left", ncols=3)
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
        in_unit, utd = gmm.get_training_data(unit_id)

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
        _, in_unit_, z = gmm.split_features(unit_id)
        in_unit, utd = gmm.get_training_data(unit_id, in_unit=in_unit_)
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
        time_range = self.time_range
        if time_range is None:
            time_range = times.min(), times.max()
        chans = torch.full(times.shape, gmm[unit_id].max_channel, device=times.device)
        waveform_channels = gmm.data.registered_waveform_channel_index[chans]
        if waveform_channels.ndim == 1:
            waveform_channels = waveform_channels[None]
        recons = gmm[unit_id].get_means(times)
        recons = gmm[unit_id].to_waveform_channels(recons, waveform_channels=waveform_channels)
        recons = gmm.data.tpca.inverse_transform(recons, chans, channel_index=gmm.data.registered_waveform_channel_index)
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
            channel_index=gmm.data.registered_waveform_channel_index.numpy(force=True),
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
    width = 2
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
        time_range = self.time_range
        if time_range is None:
            time_range = times.min(), times.max()
        chans = torch.full(times.shape, gmm[unit_id].max_channel, device=times.device).squeeze()
        waveform_channels = gmm.data.registered_waveform_channel_index[chans].to(gmm.device)
        if waveform_channels.ndim == 1:
            waveform_channels = waveform_channels[None]
        recons = gmm[unit_id].get_means(times).to(gmm.device)
        recons = gmm[unit_id].to_waveform_channels(recons, waveform_channels=waveform_channels)

        main_channel = gmm[unit_id].max_channel
        nixs, cixs = torch.nonzero(waveform_channels == main_channel, as_tuple=True)
        recons = torch.take_along_dim(recons, cixs[:, None, None], dim=2)[:, :, 0]

        recons = gmm.data.tpca._inverse_transform_in_probe(recons)
        if gmm[unit_id].do_interp:
            c = times.numpy(force=True)
            c = (c - time_range[0]) / (time_range[1] - time_range[0])
            colors = self.cmap(c)
        else:
            colors = "r"

        recons = recons.numpy(force=True)
        chans = chans.numpy(force=True)

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
    width = 2
    height = 2

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
        chans = gmm.data.channels[in_unit[wh]]
        waveforms = utd["waveforms"][wh]
        waveform_channels = utd["waveform_channels"][wh]

        main_channel = gmm[unit_id].max_channel
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
        in_unit, utd = gmm.get_training_data(unit_id)
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

        main_channel = gmm[unit_id].max_channel
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
                axis.text(i, j, f"{label:.2f}", ha="center", va="center")
        plt.colorbar(im, ax=axis, shrink=0.3)
        axis.set_xticks(range(len(times)), [f"{t:0.1f}" for t in times])
        axis.set_yticks(range(len(times)), [f"{t:0.1f}" for t in times])
        axis.set_title(self.title)


default_gmm_plots = (
    ISIHistogram(),
    GridMeansSingleChanPlot(),
    InputWaveformsSingleChanPlot(),
    ResidualsSingleChanPlot(),
    AmplitudesOverTimePlot(),
    BadnessesOverTimePlot(),
    EmbedsOverTimePlot(),
    DPCSplitPlot(kind="residual"),
    DPCSplitPlot(kind="global"),
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
    figsize=(15, 10),
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