import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from .. import config
from ..templates import templates
from ..util import data_util, spike_features
from . import analysis_plots, scatterplots

basic_template_config = config.TemplateConfig(
    realign_peaks=False, superres_templates=False
)


def sorting_scatter_animation(
    sorting_analysis,
    chunk_sortings=None,
    chunk_templates=None,
    chunk_time_ranges_s=None,
    chunk_length_samples=None,
    show_template_main_traces=True,
    scatter_template_features=True,
    depth_feature="z",
    features=("x", "amplitude", "log_peak_to_trough"),
    template_config=basic_template_config,
    trough_offset_samples=42,
    spike_length_samples=121,
    n_jobs_templates=0,
    device=None,
    feature_getters=None,
    interval=200,
    figsize=(11, 8.5),
):
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
        interval=interval,
        figsize=figsize,
    )


usual_feature_getters = {
    "x": ("point_source_localizations", (slice(None), 0)),
    "z": ("point_source_localizations", (slice(None), 2)),
    "amplitude": ("denoised_ptp_amplitudes", slice(None)),
    "log_peak_to_trough": ("denoised_logpeaktotrough", slice(None)),
}


def get_template_features(template_data, feature_names, scatter_template_features):
    if not scatter_template_features:
        return {}

    features = {}
    if "x" in feature_names or "z" in feature_names:
        xyza = template_data.template_locations()
        features["x"] = xyza["x"]
        features["z"] = xyza["z_abs"]

    if "amplitude" in feature_names:
        features["amplitude"] = template_data.templates.ptp(1).max(1)

    if "log_peak_to_trough" in feature_names:
        features["log_peak_to_trough"] = spike_features.peak_to_trough(
            template_data.templates
        )

    return features


def update_frame(
    axes,
    feature_scatters,
    template_scatters,
    chunk_time_range,
    chunk_sorting,
    all_spike_features,
    depth_feature="z",
    chunk_template_data=None,
    max_n_templates=None,
    chunk_template_features=None,
    show_template_main_traces=True,
):
    # main traces
    if show_template_main_traces and chunk_template_data is not None:
        axes[0].clear()
        analysis_plots.scatter_max_channel_waveforms(axes[0], chunk_template_data, lw=1)
        axes = axes[1:]

    spike_mask = np.equal(
        chunk_sorting.times_seconds,
        chunk_sorting.times_seconds.clip(*chunk_time_range),
    )
    non_depth_keys = [k for k in all_spike_features if k != depth_feature]
    for j, k in enumerate(non_depth_keys):
        # determine this chunk's spike mask
        feature_scatters[k] = scatterplots.scatter_feature_vs_depth(
            all_spike_features[k][spike_mask],
            all_spike_features[depth_feature][spike_mask],
            labels=chunk_sorting.labels[spike_mask],
            pad_to_max=True,
            scat=feature_scatters.get(k, None),
            geom=chunk_template_data.registered_geom,
        )
        if chunk_template_features is not None:
            template_scatters[k] = scatterplots.scatter_feature_vs_depth(
                chunk_template_features[k],
                chunk_template_features[depth_feature],
                labels=chunk_template_data.unit_ids,
                pad_to_max=True,
                max_spikes_plot=max_n_templates,
                scat=template_scatters.get(k, None),
                s=5,
                linewidth=1,
                edgecolor="k",
                geom=chunk_template_data.registered_geom,
                rasterized=False,
            )

        xmin, xmax = all_spike_features[k].min(), all_spike_features[k].max()
        margin = (xmax - xmin) * 0.025
        axes[k].set_xlim([xmin - margin, xmax + margin])


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
    interval=200,
    figsize=(11, 8.5),
):
    if feature_getters is None:
        feature_getters = usual_feature_getters

    if len(feature_names):
        with h5py.File(sorting_analysis.hdf5_path, "r", locking=False) as h5:
            all_spike_features = {
                name: h5[feature_getters[name][0]][feature_getters[name][1]]
                for name in feature_names
            }
            if (
                depth_feature in all_spike_features
                and sorting_analysis.motion_est is not None
            ):
                all_spike_features[depth_feature] = (
                    sorting_analysis.motion_est.correct_s(
                        h5["times_seconds", :],
                        depths_um=all_spike_features[depth_feature],
                    )
                )
    else:
        all_spike_features = {}

    all_template_features = None
    max_n_templates = max(t.templates.shape[0] for t in chunk_templates)
    if chunk_templates is not None and scatter_template_features:
        all_template_features = [
            get_template_features(chunk_td, feature_names, scatter_template_features)
            for chunk_td in chunk_templates
        ]

    # create and lay out figure
    ncols = len(feature_names) + show_template_main_traces
    fig, axes = plt.subplots(
        nrows=1, ncols=ncols, figsize=figsize, layout="constrained", sharey=True
    )
    feature_scatters = {}
    template_scatters = {}

    def closure(frame):
        j = frame
        update_frame(
            axes,
            feature_scatters,
            template_scatters,
            chunk_time_ranges_s[j],
            chunk_sortings[j],
            all_spike_features,
            depth_feature=depth_feature,
            chunk_template_data=chunk_templates[j],
            max_n_templates=max_n_templates,
            chunk_template_features=all_template_features[j],
            show_template_main_traces=show_template_main_traces,
        )

    # initialize
    closure(0)

    # run
    anim = animation.FuncAnimation(
        fig, closure, interval=interval, frames=len(chunk_time_ranges_s)
    )

    return fig, anim
