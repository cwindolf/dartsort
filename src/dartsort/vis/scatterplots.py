import colorcet
import h5py
import matplotlib.pyplot as plt
import numpy as np


def scatter_spike_features(
    hdf5_filename=None,
    sorting=None,
    times_s=None,
    depths_um=None,
    x_um=None,
    amplitudes=None,
    geom=None,
    figure=None,
    axes=None,
    width_ratios=(1, 1, 3),
    semilog_amplitudes=True,
    show_geom=True,
    geom_scatter_kw=dict(s=5, color="k", lw=0),
    amplitude_color_cutoff=15,
    max_spikes_plot=500_000,
    depth_margin_um=100,
    s=1,
    linewidth=0,
    random_seed=0,
    **scatter_kw,
):
    if axes is not None:
        assert axes.size == 3
        figure = axes.flat[0]
    if figure is None:
        figure = plt.gcf()
    if axes is None:
        axes = figure.subplots(
            ncols=3, sharey=True, gridspec_kw=dict(width_ratios=width_ratios)
        )

    if hdf5_filename is not None:
        with h5py.File(hdf5_filename, "r") as h5:
            times_s = h5["times_seconds"][:]
            x = h5["point_source_localizations"][:, 0]
            depths_um = h5["point_source_localizations"][:, 2]
            amplitudes = h5["denoised_amplitudes"][:]
            geom = h5["geom"][:]

    _, s_x = scatter_x_vs_depth(
        x=x,
        depths_um=depths_um,
        amplitudes=amplitudes,
        show_geom=show_geom,
        geom_scatter_kw=geom_scatter_kw,
        sorting=sorting,
        geom=geom,
        depth_margin_um=depth_margin_um,
        ax=axes.flat[0],
        max_spikes_plot=max_spikes_plot,
        amplitude_color_cutoff=amplitude_color_cutoff,
        random_seed=random_seed,
        s=s,
        linewidth=linewidth,
        **scatter_kw,
    )

    _, s_a = scatter_amplitudes_vs_depth(
        depths_um=depths_um,
        amplitudes=amplitudes,
        semilog_amplitudes=semilog_amplitudes,
        sorting=sorting,
        geom=geom,
        depth_margin_um=depth_margin_um,
        ax=axes.flat[1],
        max_spikes_plot=max_spikes_plot,
        amplitude_color_cutoff=amplitude_color_cutoff,
        random_seed=random_seed,
        s=s,
        linewidth=linewidth,
        **scatter_kw,
    )

    _, s_t = scatter_time_vs_depth(
        times_s=times_s,
        depths_um=depths_um,
        amplitudes=amplitudes,
        sorting=sorting,
        geom=geom,
        depth_margin_um=depth_margin_um,
        ax=axes.flat[2],
        max_spikes_plot=max_spikes_plot,
        amplitude_color_cutoff=amplitude_color_cutoff,
        random_seed=random_seed,
        s=s,
        linewidth=linewidth,
        **scatter_kw,
    )

    return figure, axes, (s_x, s_a, s_t)


def scatter_time_vs_depth(
    hdf5_filename=None,
    times_s=None,
    depths_um=None,
    amplitudes=None,
    labels=None,
    sorting=None,
    geom=None,
    depth_margin_um=100,
    ax=None,
    max_spikes_plot=500_000,
    amplitude_color_cutoff=15,
    random_seed=0,
    s=1,
    linewidth=0,
    **scatter_kw,
):
    """Scatter plot of spike time vs spike depth (vertical position on probe)

    In this plot, each dot is a spike. Spike data can be passed directly via
    the times_s, depths_um, and (one of) amplitudes or labels as arrays, or
    alternatively, these can be left unset and they will be loaded from
    hdf5_filename when it is supplied.
    """
    if hdf5_filename is not None:
        with h5py.File(hdf5_filename, "r") as h5:
            times_s = h5["times_seconds"][:]
            depths_um = h5["point_source_localizations"][:, 2]
            amplitudes = h5["denoised_amplitudes"][:]
            geom = h5["geom"][:]

    return scatter_feature_vs_depth(
        times_s,
        depths_um,
        amplitudes=amplitudes,
        sorting=sorting,
        geom=geom,
        ax=ax,
        max_spikes_plot=max_spikes_plot,
        amplitude_color_cutoff=amplitude_color_cutoff,
        depth_margin_um=depth_margin_um,
        s=s,
        linewidth=linewidth,
        random_seed=random_seed,
        **scatter_kw,
    )


def scatter_x_vs_depth(
    hdf5_filename=None,
    x=None,
    depths_um=None,
    amplitudes=None,
    labels=None,
    show_geom=True,
    geom_scatter_kw=dict(s=5, color="k", lw=0),
    sorting=None,
    geom=None,
    depth_margin_um=100,
    ax=None,
    max_spikes_plot=500_000,
    amplitude_color_cutoff=15,
    random_seed=0,
    s=1,
    linewidth=0,
    **scatter_kw,
):
    """Scatter plot of spike horizontal pos vs spike depth (vertical position on probe)"""
    if hdf5_filename is not None:
        with h5py.File(hdf5_filename, "r") as h5:
            x = h5["point_source_localizations"][:, 0]
            depths_um = h5["point_source_localizations"][:, 2]
            amplitudes = h5["denoised_amplitudes"][:]
            geom = h5["geom"][:]

    ax, s1 = scatter_feature_vs_depth(
        x,
        depths_um,
        amplitudes=amplitudes,
        sorting=sorting,
        geom=geom,
        ax=ax,
        max_spikes_plot=max_spikes_plot,
        amplitude_color_cutoff=amplitude_color_cutoff,
        depth_margin_um=depth_margin_um,
        s=s,
        linewidth=linewidth,
        random_seed=random_seed,
        **scatter_kw,
    )
    if show_geom and geom is not None:
        ax.scatter(*geom.T, **geom_scatter_kw)
    return ax, s1


def scatter_amplitudes_vs_depth(
    hdf5_filename=None,
    depths_um=None,
    amplitudes=None,
    labels=None,
    semilog_amplitudes=True,
    sorting=None,
    geom=None,
    depth_margin_um=100,
    ax=None,
    max_spikes_plot=500_000,
    amplitude_color_cutoff=15,
    random_seed=0,
    s=1,
    linewidth=0,
    **scatter_kw,
):
    """Scatter plot of spike horizontal pos vs spike depth (vertical position on probe)"""
    if hdf5_filename is not None:
        with h5py.File(hdf5_filename, "r") as h5:
            depths_um = h5["point_source_localizations"][:, 2]
            amplitudes = h5["denoised_amplitudes"][:]
            geom = h5["geom"][:]

    ax, s = scatter_feature_vs_depth(
        amplitudes,
        depths_um,
        amplitudes=amplitudes,
        sorting=sorting,
        geom=geom,
        ax=ax,
        max_spikes_plot=max_spikes_plot,
        amplitude_color_cutoff=amplitude_color_cutoff,
        depth_margin_um=depth_margin_um,
        s=s,
        linewidth=linewidth,
        random_seed=random_seed,
        **scatter_kw,
    )
    if semilog_amplitudes:
        ax.semilogx()
    return ax, s


def scatter_feature_vs_depth(
    feature,
    depths_um,
    amplitudes=None,
    sorting=None,
    geom=None,
    ax=None,
    max_spikes_plot=500_000,
    amplitude_color_cutoff=15,
    depth_margin_um=100,
    s=1,
    linewidth=0,
    random_seed=0,
    **scatter_kw,
):
    assert feature.shape == depths_um.shape
    if amplitudes is not None:
        assert feature.shape == amplitudes.shape
    assert feature.ndim == 1

    if ax is None:
        ax = plt.gca()

    # subset spikes according to margin and sampling
    n_spikes = len(feature)
    to_show = np.arange(n_spikes)
    if geom is not None:
        to_show = np.flatnonzero(
            (depths_um > geom[:, 1].min() - depth_margin_um)
            & (depths_um < geom[:, 1].max() + depth_margin_um)
        )
    if len(to_show) > max_spikes_plot:
        rg = np.random.default_rng(random_seed)
        to_show = rg.choice(to_show, size=max_spikes_plot, replace=False)

    # order by amplitude so that high amplitude units show up
    if amplitudes is not None:
        to_show = to_show[np.argsort(amplitudes[to_show])]

    if sorting is not None:
        labels = sorting.labels
    if labels is None:
        c = np.clip(amplitudes[to_show], 0, amplitude_color_cutoff)
        cmap = plt.cm.viridis
    else:
        c = labels
        cmap = colorcet.m_glasbey_light
        kept = labels[to_show] >= 0
        ax.scatter(
            feature[to_show[~kept]],
            depths_um[to_show[~kept]],
            color="dimgray",
            s=s,
            linewidth=linewidth,
            **scatter_kw,
        )
        to_show = to_show[kept]

    s = ax.scatter(
        feature[to_show],
        depths_um[to_show],
        c=c[to_show],
        cmap=cmap,
        s=s,
        linewidth=linewidth,
        **scatter_kw,
    )
    return ax, s
