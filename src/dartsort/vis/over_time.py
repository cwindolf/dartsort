from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from .. import config
from ..cluster import merge
from ..templates import templates
from ..templates.get_templates import fit_tsvd
from ..util import data_util, spike_features, analysis
from . import analysis_plots, scatterplots, layout, unit

basic_template_config = config.TemplateConfig(
    realign_peaks=False, superres_templates=False
)


# -- over_time_summary stuff


class OverTimePlot(layout.BasePlot):
    width = 8
    height = 1
    kind = "over_time"

    def draw(self, panel, chunk_sorting_analyses, unit_id):
        raise NotImplementedError


class UnitPlotOverTime(OverTimePlot):
    def __init__(self, unit_plot, height=1, sharey=True):
        self.height = height
        self.unit_plot = unit_plot
        self.sharey = sharey and unit_plot.can_sharey

    def notify_global_params(self, **params):
        super().notify_global_params(**params)
        self.unit_plot.notify_global_params(**params)

    def draw(self, panel, sorting_analysis, unit_id):
        chunk_sorting_analyses = sorting_analysis
        axes = panel.subplots(
            ncols=len(chunk_sorting_analyses), sharey=self.sharey
        )
        first = True
        for j, (ax, csa) in enumerate(zip(axes, chunk_sorting_analyses)):
            if unit_id not in csa.unit_ids:
                ax.axis("off")
                continue
            self.unit_plot.draw(None, csa, unit_id, axis=ax)
            if self.sharey:
                ax.set_yticks([])
                ax.set_ylabel("")
            if not first:
                lg = ax.get_legend()
                if lg is not None:
                    lg.remove()
            first = False


class SelfDistanceOverTime(OverTimePlot):
    height = 1.5

    def __init__(self, show_values=True):
        super().__init__()
        self.show_values = show_values

    def draw(self, panel, sorting_analysis, unit_id):
        chunk_sorting_analyses = sorting_analysis
        imax, longax = panel.subplots(ncols=2, width_ratios=[1, 2])
        nchunks = len(chunk_sorting_analyses)

        temps = []
        counts = []
        chunkixs = []
        for j, csa in enumerate(chunk_sorting_analyses):
            ctd = csa.coarse_template_data
            um = ctd.unit_mask(unit_id)
            if not um.sum():
                continue
            um = np.flatnonzero(um)
            assert um.size == 1
            um = um[0]
            temps.append(ctd.templates[um])
            counts.append(ctd.spike_counts[um])
            chunkixs.append(j)
        temps = np.stack(temps, axis=0)
        counts = np.ravel(counts)
        chunkixs = np.array(chunkixs)
        chunks_td = templates.TemplateData(temps, chunkixs, counts)
        chunks, dists, _, _ = merge.calculate_merge_distances(
            chunks_td, n_jobs=0
        )

        im = imax.imshow(
            dists,
            vmin=0,
            vmax=1,
            cmap=plt.cm.RdGy,
            origin="lower",
            interpolation="none",
        )
        imax.set_xticks(range(len(chunkixs)), chunkixs)
        imax.set_yticks(range(len(chunkixs)), chunkixs)
        if self.show_values:
            for (j, i), label in np.ndenumerate(dists):
                imax.text(i, j, f"{label:.2f}", ha="center", va="center")
        plt.colorbar(im, ax=imax, shrink=0.3)
        imax.set_title("inter-chunk template distances")

        longax.plot(chunkixs[:-1], np.diag(dists, k=1))
        longax.set_ylabel("dist between temps of chunk $i,i+1$")
        longax.set_xlabel("chunk index $i$")
        longax.set_title("neighboring chunk template distances")


over_time_plots = (
    UnitPlotOverTime(unit.RawWaveformPlot(), height=2),
    SelfDistanceOverTime(),
    UnitPlotOverTime(unit.PCAScatter()),
    UnitPlotOverTime(unit.XZScatter()),
    UnitPlotOverTime(unit.TimeAmpScatter()),
    UnitPlotOverTime(unit.NearbyCoarseTemplatesPlot(), height=2),
    UnitPlotOverTime(unit.CoarseTemplateDistancePlot(), height=1.5),
)


def make_all_over_time_summaries(
    sorting_analysis,
    save_folder,
    plots=over_time_plots,
    chunk_length_s=300.0,
    template_config=basic_template_config,
    channel_show_radius_um=50.0,
    amplitude_color_cutoff=15.0,
    pca_radius_um=75.0,
    max_height=None,
    figsize=(16, 8.5),
    hspace=0.05,
    dpi=200,
    image_ext="png",
    n_jobs=0,
    n_jobs_templates=0,
    show_progress=True,
    overwrite=False,
    unit_ids=None,
    analysis_kw=None,
):
    # get unit ids
    if unit_ids is None:
        unit_ids = sorting_analysis.unit_ids

    # check if done before expensive stuff
    done = not overwrite and unit.all_summaries_done(unit_ids, save_folder)
    if done:
        return

    # make chunk sorting analyses
    # chunk sortings
    chunk_time_ranges_s, chunk_sortings = data_util.time_chunk_sortings(
        sorting=sorting_analysis.sorting,
        recording=sorting_analysis.recording,
        chunk_length_samples=chunk_length_s
        * sorting_analysis.recording.sampling_frequency,
    )
    # tsvd
    tsvd = fit_tsvd(
        sorting_analysis.recording,
        sorting_analysis.sorting,
        denoising_fit_radius=pca_radius_um,
    )
    # chunk sorting analyses
    if analysis_kw is None:
        analysis_kw = {}
    chunk_sorting_analyses = [
        analysis.DARTsortAnalysis.from_sorting(
            sorting_analysis.recording,
            chunk_sorting,
            motion_est=sorting_analysis.motion_est,
            template_config=template_config,
            allow_template_reload=False,
            n_jobs_templates=n_jobs_templates,
            denoising_tsvd=tsvd,
            **analysis_kw,
        )
        for chunk_sorting in chunk_sortings
    ]

    if max_height is None:
        max_height = sum(p.height for p in plots)
    unit.make_all_summaries(
        chunk_sorting_analyses,
        save_folder,
        plots=plots,
        channel_show_radius_um=channel_show_radius_um,
        amplitude_color_cutoff=amplitude_color_cutoff,
        pca_radius_um=pca_radius_um,
        max_height=max_height,
        figsize=figsize,
        hspace=hspace,
        dpi=dpi,
        image_ext=image_ext,
        n_jobs=n_jobs,
        show_progress=show_progress,
        overwrite=overwrite,
        unit_ids=unit_ids,
    )


# -- animation stuff


def sorting_scatter_animation(
    sorting_analysis,
    output_path,
    chunk_sortings=None,
    chunk_templates=None,
    chunk_time_ranges_s=None,
    chunk_length_samples=None,
    show_template_main_traces=True,
    scatter_template_features=True,
    suptitle=None,
    depth_feature="z",
    features=("x", "amplitude", "log_peak_to_trough"),
    template_config=basic_template_config,
    templates_tsvd=None,
    trough_offset_samples=42,
    spike_length_samples=121,
    n_jobs_templates=0,
    device=None,
    feature_getters=None,
    interval=1500,
    figsize=(11, 8.5),
    dpi=150,
):
    output_path = Path(output_path)
    assert output_path.parent.exists()
    assert output_path.name.endswith(".mp4")

    if chunk_templates is not None:
        assert chunk_sortings is not None

    # first, what sortings are we using in each chunk?
    if chunk_sortings is None:
        chunk_time_ranges_s, chunk_sortings = data_util.time_chunk_sortings(
            sorting=sorting_analysis.sorting,
            recording=sorting_analysis.recording,
            chunk_time_ranges_s=chunk_time_ranges_s,
            chunk_length_samples=chunk_length_samples,
        )

    # next, if we need them, make templates for each unit in each chunk
    use_chunk_templates = show_template_main_traces or scatter_template_features
    if use_chunk_templates and chunk_templates is None:
        chunk_templates = templates.get_chunked_templates(
            sorting_analysis.recording,
            template_config,
            global_sorting=sorting_analysis.sorting,
            chunk_sortings=chunk_sortings,
            chunk_time_ranges_s=chunk_time_ranges_s,
            motion_est=sorting_analysis.motion_est,
            n_jobs=n_jobs_templates,
            device=device,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            tsvd=templates_tsvd,
        )

    # make the animation
    fig, anim = make_animation(
        sorting_analysis,
        chunk_time_ranges_s,
        chunk_sortings,
        chunk_templates,
        feature_names=features,
        depth_feature=depth_feature,
        show_template_main_traces=show_template_main_traces,
        scatter_template_features=scatter_template_features,
        feature_getters=None,
        suptitle=suptitle,
        interval=interval,
        figsize=figsize,
    )
    # messing with this to try to make the videos work
    # with mac quicktime
    anim.save(
        str(output_path),
        writer="ffmpeg",
        codec="libx264",
        extra_args=[
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            # "-c:v", "hevc_videotoolbox",
            # "-level:v", "5.1", "-profile:v", "High",
        ],
        dpi=dpi,
        bitrate=-1,
    )
    plt.close(fig)


usual_feature_getters = {
    "x": ("point_source_localizations", (slice(None), 0)),
    "z": ("point_source_localizations", (slice(None), 2)),
    "amplitude": ("denoised_ptp_amplitudes", slice(None)),
    "log_peak_to_trough": ("denoised_logpeaktotrough", slice(None)),
}


def get_template_features(
    template_data, feature_names, scatter_template_features
):
    if not scatter_template_features:
        return {}

    features = {}
    if "x" in feature_names or "z" in feature_names:
        xyza = template_data.template_locations()
        features["x"] = xyza["x"]
        features["z"] = xyza["z_abs"]

    if "amplitude" in feature_names:
        features["amplitude"] = np.ptp(template_data.templates, 1).max(1)

    if "log_peak_to_trough" in feature_names:
        features["log_peak_to_trough"] = spike_features.peak_to_trough(
            template_data.templates
        )

    return features


def update_frame(
    axes,
    feature_scatters,
    feature_ellips,
    template_scatters,
    geom_scatter,
    chunk_time_range,
    chunk_sorting,
    all_spike_features,
    geom,
    motion_est,
    depth_feature="z",
    semilog_amplitude=True,
    chunk_template_data=None,
    max_n_templates=None,
    chunk_template_features=None,
    show_template_main_traces=True,
):
    # main traces
    if show_template_main_traces and chunk_template_data is not None:
        axes[0].clear()
        analysis_plots.scatter_max_channel_waveforms(
            axes[0], chunk_template_data, lw=1
        )
        xmin, dx = geom[:, 0].min(), np.ptp(geom[:, 0])
        axes[0].set_xlim([xmin - dx / 2, xmin + 3 * dx / 2])
        axes = axes[1:]

    spike_mask = np.equal(
        chunk_sorting.times_seconds,
        chunk_sorting.times_seconds.clip(*chunk_time_range),
    )
    non_depth_keys = [k for k in all_spike_features if k != depth_feature]
    for j, k in enumerate(non_depth_keys):
        # determine this chunk's spike mask
        (
            _,
            feature_scatters[k],
            feature_ellips[k],
        ) = scatterplots.scatter_feature_vs_depth(
            all_spike_features[k][spike_mask],
            all_spike_features[depth_feature][spike_mask],
            ax=axes[j],
            labels=chunk_sorting.labels[spike_mask],
            pad_to_max=True,
            scat=feature_scatters.get(k, None),
            geom=chunk_template_data.registered_geom,
            show_ellipses=True,
            max_n_labels=max_n_templates,
            ellip=feature_ellips.get(k, None),
            show_triaged=False,
        )

        if k != "x":
            xmin, xmax = (
                all_spike_features[k].min(),
                all_spike_features[k].max(),
            )
            margin = (xmax - xmin) * 0.025
        else:
            xmin, xmax = geom[:, 0].min(), geom[:, 0].max()
            margin = 100
            disp = motion_est.disp_at_s(
                sum(chunk_time_range) / 2, geom[:, 1], grid=True
            )
            disp = np.c_[np.zeros_like(disp), disp]
            if geom_scatter:
                geom_scatter[0].set_offsets(geom - disp)
            else:
                geom_scatter.append(
                    axes[j].scatter(
                        *(geom - disp).T, s=5, marker="s", color="k", lw=0
                    )
                )

        if chunk_template_features is not None:
            _, template_scatters[k] = scatterplots.scatter_feature_vs_depth(
                chunk_template_features[k],
                chunk_template_features[depth_feature],
                ax=axes[j],
                labels=chunk_template_data.unit_ids,
                pad_to_max=True,
                max_spikes_plot=max_n_templates,
                scat=template_scatters.get(k, None),
                s=15,
                linewidth=1,
                edgecolor="k",
                marker="s",
                geom=chunk_template_data.registered_geom,
                rasterized=False,
                show_triaged=False,
            )

        if k == "amplitude" and semilog_amplitude:
            margin = 1.05
            axes[j].set_xlim([xmin / margin, xmax * margin])
            axes[j].semilogx()
        else:
            axes[j].set_xlim([xmin - margin, xmax + margin])


def make_animation(
    sorting_analysis,
    chunk_time_ranges_s,
    chunk_sortings,
    chunk_templates,
    feature_names,
    depth_feature="z",
    show_template_main_traces=True,
    scatter_template_features=True,
    feature_getters=None,
    suptitle=None,
    interval=200,
    figsize=(11, 8.5),
):
    if feature_getters is None:
        feature_getters = usual_feature_getters

    if len(feature_names):
        with h5py.File(sorting_analysis.hdf5_path, "r", locking=False) as h5:
            all_spike_features = {
                name: h5[feature_getters[name][0]][feature_getters[name][1]]
                for name in list(feature_names) + [depth_feature]
            }
            if (
                depth_feature in all_spike_features
                and sorting_analysis.motion_est is not None
            ):
                all_spike_features[
                    depth_feature
                ] = sorting_analysis.motion_est.correct_s(
                    t_s=h5["times_seconds"][()],
                    depth_um=all_spike_features[depth_feature],
                )
    else:
        all_spike_features = {}

    all_template_features = None
    max_n_templates = max(t.templates.shape[0] for t in chunk_templates)
    if chunk_templates is not None and scatter_template_features:
        all_template_features = [
            get_template_features(
                chunk_td,
                list(feature_names) + [depth_feature],
                scatter_template_features,
            )
            for chunk_td in chunk_templates
        ]

    # create and lay out figure
    ncols = len(feature_names) + show_template_main_traces
    fig, axes = plt.subplots(
        nrows=1, ncols=ncols, figsize=figsize, layout="constrained", sharey=True
    )
    feature_scatters = {}
    feature_ellips = {}
    template_scatters = {}
    geom_scatter = []

    def closure(frame):
        j = frame
        print(f"animation {frame=}")
        update_frame(
            axes,
            feature_scatters,
            feature_ellips,
            template_scatters,
            geom_scatter,
            chunk_time_ranges_s[j],
            chunk_sortings[j],
            all_spike_features,
            sorting_analysis.recording.get_channel_locations(),
            sorting_analysis.motion_est,
            depth_feature=depth_feature,
            chunk_template_data=chunk_templates[j],
            max_n_templates=max_n_templates,
            chunk_template_features=all_template_features[j],
            show_template_main_traces=show_template_main_traces,
        )

    # initialize
    closure(0)
    if suptitle is not None:
        fig.suptitle(suptitle)
    for j, k in enumerate(feature_names):
        axes[j + show_template_main_traces].set_xlabel(k)

    # run
    anim = animation.FuncAnimation(
        fig, closure, interval=interval, frames=len(chunk_time_ranges_s)
    )

    return fig, anim
