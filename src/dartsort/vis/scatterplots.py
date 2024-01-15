import colorcet
import h5py
import matplotlib.pyplot as plt
import numpy as np


def scatter_spike_features(
    hdf5_filename=None,
    sorting=None,
    motion_est=None,
    registered=False,
    times_s=None,
    depths_um=None,
    x=None,
    amplitudes=None,
    geom=None,
    figure=None,
    axes=None,
    width_ratios=(1, 1, 3),
    semilog_amplitudes=True,
    show_geom=True,
    geom_scatter_kw=dict(s=5, marker="s", color="k", lw=0),
    amplitude_color_cutoff=15,
    amplitude_cmap=plt.cm.viridis,
    max_spikes_plot=500_000,
    probe_margin_um=100,
    t_min=-np.inf,
    t_max=np.inf,
    s=1,
    linewidth=0,
    limits="probe_margin",
    label_axes=True,
    random_seed=0,
    figsize=None,
    **scatter_kw,
):
    """3-axis scatter plot of spike depths vs. horizontal pos, amplitude, and time

    Returns
    -------
    figure, axes, (s_x, s_a, s_t)
        Matplotlib figure, axes array, and 3-tuple of scatterplot PathCollections
    """
    if axes is not None:
        assert axes.size == 3
        figure = axes.flat[0].figure
    if figure is None:
        figure = plt.gcf()
    if axes is None:
        axes = figure.subplots(
            ncols=3,
            sharey=True,
            gridspec_kw=dict(width_ratios=width_ratios),
            figsize=figsize,
        )

    if hdf5_filename is not None:
        with h5py.File(hdf5_filename, "r") as h5:
            times_s = h5["times_seconds"][:]
            x = h5["point_source_localizations"][:, 0]
            depths_um = h5["point_source_localizations"][:, 2]
            amplitudes = h5["denoised_amplitudes"][:]
            geom = h5["geom"][:]

    to_show = np.flatnonzero(np.clip(times_s, t_min, t_max) == times_s)
    if geom is not None:
        to_show = to_show[
            (depths_um[to_show] > geom[:, 1].min() - probe_margin_um)
            & (depths_um[to_show] < geom[:, 1].max() + probe_margin_um)
            & (x[to_show] > geom[:, 0].min() - probe_margin_um)
            & (x[to_show] < geom[:, 0].max() + probe_margin_um)
        ]

    _, s_x = scatter_x_vs_depth(
        x=x,
        depths_um=depths_um,
        times_s=times_s,
        amplitudes=amplitudes,
        show_geom=show_geom,
        geom_scatter_kw=geom_scatter_kw,
        sorting=sorting,
        motion_est=motion_est,
        registered=registered,
        geom=geom,
        probe_margin_um=probe_margin_um,
        ax=axes.flat[0],
        max_spikes_plot=max_spikes_plot,
        amplitude_color_cutoff=amplitude_color_cutoff,
        amplitude_cmap=amplitude_cmap,
        random_seed=random_seed,
        s=s,
        linewidth=linewidth,
        to_show=to_show,
        **scatter_kw,
    )

    _, s_a = scatter_amplitudes_vs_depth(
        depths_um=depths_um,
        amplitudes=amplitudes,
        times_s=times_s,
        semilog_amplitudes=semilog_amplitudes,
        sorting=sorting,
        motion_est=motion_est,
        registered=registered,
        geom=geom,
        probe_margin_um=probe_margin_um,
        ax=axes.flat[1],
        max_spikes_plot=max_spikes_plot,
        amplitude_color_cutoff=amplitude_color_cutoff,
        amplitude_cmap=amplitude_cmap,
        random_seed=random_seed,
        s=s,
        linewidth=linewidth,
        to_show=to_show,
        **scatter_kw,
    )

    _, s_t = scatter_time_vs_depth(
        times_s=times_s,
        depths_um=depths_um,
        amplitudes=amplitudes,
        sorting=sorting,
        motion_est=motion_est,
        registered=registered,
        geom=geom,
        probe_margin_um=probe_margin_um,
        ax=axes.flat[2],
        max_spikes_plot=max_spikes_plot,
        amplitude_color_cutoff=amplitude_color_cutoff,
        amplitude_cmap=amplitude_cmap,
        random_seed=random_seed,
        s=s,
        linewidth=linewidth,
        to_show=to_show,
        **scatter_kw,
    )

    if label_axes:
        axes[0].set_ylabel(("registered " * registered) + "depth (um)")
        axes[0].set_xlabel("x (um)")
        axes[1].set_xlabel("amplitude (su)")
        axes[2].set_xlabel("time (s)")

    return figure, axes, (s_x, s_a, s_t)


def scatter_time_vs_depth(
    hdf5_filename=None,
    sorting=None,
    motion_est=None,
    registered=False,
    times_s=None,
    depths_um=None,
    amplitudes=None,
    labels=None,
    geom=None,
    probe_margin_um=100,
    ax=None,
    max_spikes_plot=500_000,
    amplitude_color_cutoff=15,
    amplitude_cmap=plt.cm.viridis,
    limits="probe_margin",
    random_seed=0,
    s=1,
    linewidth=0,
    to_show=None,
    **scatter_kw,
):
    """Scatter plot of spike time vs spike depth (vertical position on probe)

    In this plot, each dot is a spike. Spike data can be passed directly via
    the times_s, depths_um, and (one of) amplitudes or labels as arrays, or
    alternatively, these can be left unset and they will be loaded from
    hdf5_filename when it is supplied.

    Returns: axis, scatter
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
        times_s=times_s,
        amplitudes=amplitudes,
        sorting=sorting,
        motion_est=motion_est,
        registered=registered,
        geom=geom,
        ax=ax,
        max_spikes_plot=max_spikes_plot,
        amplitude_color_cutoff=amplitude_color_cutoff,
        amplitude_cmap=amplitude_cmap,
        probe_margin_um=probe_margin_um,
        s=s,
        linewidth=linewidth,
        random_seed=random_seed,
        to_show=to_show,
        **scatter_kw,
    )


def scatter_x_vs_depth(
    hdf5_filename=None,
    sorting=None,
    motion_est=None,
    registered=False,
    x=None,
    depths_um=None,
    amplitudes=None,
    times_s=None,
    labels=None,
    show_geom=True,
    geom_scatter_kw=dict(s=5, marker="s", color="k", lw=0),
    geom=None,
    probe_margin_um=100,
    ax=None,
    max_spikes_plot=500_000,
    amplitude_color_cutoff=15,
    amplitude_cmap=plt.cm.viridis,
    limits="probe_margin",
    random_seed=0,
    s=1,
    linewidth=0,
    to_show=None,
    **scatter_kw,
):
    """Scatter plot of spike horizontal pos vs spike depth (vertical position on probe)"""
    if hdf5_filename is not None:
        with h5py.File(hdf5_filename, "r") as h5:
            times_s = h5["times_seconds"][:]
            x = h5["point_source_localizations"][:, 0]
            depths_um = h5["point_source_localizations"][:, 2]
            amplitudes = h5["denoised_amplitudes"][:]
            geom = h5["geom"][:]

    if to_show is None and geom is not None:
        to_show = np.flatnonzero(
            (x > geom[:, 0].min() - probe_margin_um)
            & (x < geom[:, 0].max() + probe_margin_um)
        )

    ax, s1 = scatter_feature_vs_depth(
        x,
        depths_um,
        times_s=times_s,
        amplitudes=amplitudes,
        sorting=sorting,
        motion_est=motion_est,
        registered=registered,
        geom=geom,
        ax=ax,
        max_spikes_plot=max_spikes_plot,
        amplitude_color_cutoff=amplitude_color_cutoff,
        amplitude_cmap=amplitude_cmap,
        probe_margin_um=probe_margin_um,
        s=s,
        linewidth=linewidth,
        random_seed=random_seed,
        to_show=to_show,
        **scatter_kw,
    )
    if show_geom and geom is not None:
        ax.scatter(*geom.T, **geom_scatter_kw)
    if limits == "probe_margin" and geom is not None:
        ax.set_xlim(
            [geom[:, 0].min() - probe_margin_um, geom[:, 0].max() + probe_margin_um]
        )
    return ax, s1


def scatter_amplitudes_vs_depth(
    hdf5_filename=None,
    sorting=None,
    motion_est=None,
    registered=False,
    depths_um=None,
    amplitudes=None,
    times_s=None,
    labels=None,
    semilog_amplitudes=True,
    geom=None,
    probe_margin_um=100,
    ax=None,
    max_spikes_plot=500_000,
    amplitude_color_cutoff=15,
    amplitude_cmap=plt.cm.viridis,
    limits="probe_margin",
    random_seed=0,
    s=1,
    linewidth=0,
    to_show=None,
    **scatter_kw,
):
    """Scatter plot of spike horizontal pos vs spike depth (vertical position on probe)"""
    if hdf5_filename is not None:
        with h5py.File(hdf5_filename, "r") as h5:
            times_s = h5["times_seconds"][:]
            depths_um = h5["point_source_localizations"][:, 2]
            amplitudes = h5["denoised_amplitudes"][:]
            geom = h5["geom"][:]

    ax, s = scatter_feature_vs_depth(
        amplitudes,
        depths_um,
        times_s=times_s,
        amplitudes=amplitudes,
        sorting=sorting,
        motion_est=motion_est,
        registered=registered,
        geom=geom,
        ax=ax,
        max_spikes_plot=max_spikes_plot,
        amplitude_color_cutoff=amplitude_color_cutoff,
        amplitude_cmap=amplitude_cmap,
        probe_margin_um=probe_margin_um,
        s=s,
        linewidth=linewidth,
        random_seed=random_seed,
        to_show=to_show,
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
    times_s=None,
    labels=None,
    motion_est=None,
    registered=False,
    geom=None,
    ax=None,
    max_spikes_plot=500_000,
    amplitude_color_cutoff=15,
    amplitude_cmap=plt.cm.viridis,
    probe_margin_um=100,
    s=1,
    linewidth=0,
    limits="probe_margin",
    random_seed=0,
    to_show=None,
    rasterized=True,
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
    if to_show is None:
        to_show = np.arange(n_spikes)
    if geom is not None:
        to_show = to_show[
            (depths_um[to_show] > geom[:, 1].min() - probe_margin_um)
            & (depths_um[to_show] < geom[:, 1].max() + probe_margin_um)
        ]
    if len(to_show) > max_spikes_plot:
        rg = np.random.default_rng(random_seed)
        to_show = rg.choice(to_show, size=max_spikes_plot, replace=False)

    if registered:
        assert motion_est is not None
        assert times_s is not None
        depths_um = motion_est.correct_s(times_s, depths_um)

    # order by amplitude so that high amplitude units show up
    if amplitudes is not None:
        to_show = to_show[np.argsort(amplitudes[to_show])]

    if sorting is not None:
        labels = sorting.labels
    if labels is None:
        c = np.clip(amplitudes, 0, amplitude_color_cutoff)
        cmap = amplitude_cmap
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
            rasterized=rasterized,
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
        rasterized=rasterized,
        **scatter_kw,
    )
    if limits == "probe_margin" and geom is not None:
        ax.set_ylim(
            [geom[:, 1].min() - probe_margin_um, geom[:, 1].max() + probe_margin_um]
        )
    return ax, s
