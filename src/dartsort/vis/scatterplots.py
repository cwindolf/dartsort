import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from matplotlib.patches import Ellipse
from scipy.spatial import KDTree

from .colors import glasbey1024, gray
from dartsort.cluster.density import kdtree_inliers


def scatter_spike_features(
    hdf5_filename=None,
    sorting=None,
    motion_est=None,
    registered=False,
    times_s=None,
    depths_um=None,
    x=None,
    amplitudes=None,
    labels=None,
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
    amplitudes_dataset_name="denoised_ptp_amplitudes",
    localizations_dataset_name="point_source_localizations",
    extra_features=None,
    show_triaged=True,
    remove_outliers=True,
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
    if extra_features is None:
        extra_features = {}
    if axes is None:
        axes = figure.subplots(
            ncols=3 + len(extra_features),
            sharey=True,
            gridspec_kw=dict(width_ratios=width_ratios),
        )

    if sorting is not None:
        if times_s is None:
            times_s = getattr(sorting, "times_seconds", None)
        if x is None:
            x = getattr(sorting, localizations_dataset_name, None)
            if x is not None:
                depths_um = x[:, 2]
                x = x[:, 0]
        if amplitudes is None:
            amplitudes = getattr(sorting, amplitudes_dataset_name, None)
        if hdf5_filename is None:
            hdf5_filename = sorting.parent_h5_path

    needs_load = any(
        v is None for v in (times_s, x, depths_um, amplitudes, geom)
    )
    if needs_load and hdf5_filename is not None:
        with h5py.File(hdf5_filename, "r", locking=False) as h5:
            if times_s is None:
                times_s = h5["times_seconds"][:]
            if x is None:
                x = h5[localizations_dataset_name][:, 0]
            if depths_um is None:
                depths_um = h5[localizations_dataset_name][:, 2]
            if amplitudes is None:
                amplitudes = h5[amplitudes_dataset_name][:]
            if geom is None:
                geom = h5["geom"][:]

    to_show = np.flatnonzero(np.clip(times_s, t_min, t_max) == times_s)
    if x is not None:
        to_show = to_show[np.isfinite(x[to_show])]
    if depths_um is not None:
        to_show = to_show[np.isfinite(depths_um[to_show])]
    if geom is not None:
        to_show = to_show[
            (depths_um[to_show] > geom[:, 1].min() - probe_margin_um)
            & (depths_um[to_show] < geom[:, 1].max() + probe_margin_um)
            & (x[to_show] > geom[:, 0].min() - probe_margin_um)
            & (x[to_show] < geom[:, 0].max() + probe_margin_um)
        ]
    if remove_outliers:
        a = 50 * np.log(amplitudes + 5)
        inliers, kdtree = kdtree_inliers(np.c_[depths_um, x, a][to_show])
        to_show = to_show[inliers]

    _, s_x = scatter_x_vs_depth(
        x=x,
        depths_um=depths_um,
        times_s=times_s,
        amplitudes=amplitudes,
        labels=labels,
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
        limits=limits,
        linewidth=linewidth,
        to_show=to_show,
        amplitudes_dataset_name=amplitudes_dataset_name,
        localizations_dataset_name=localizations_dataset_name,
        show_triaged=show_triaged,
        **scatter_kw,
    )

    _, s_a = scatter_amplitudes_vs_depth(
        depths_um=depths_um,
        amplitudes=amplitudes,
        labels=labels,
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
        limits=limits,
        linewidth=linewidth,
        to_show=to_show,
        amplitudes_dataset_name=amplitudes_dataset_name,
        localizations_dataset_name=localizations_dataset_name,
        show_triaged=show_triaged,
        **scatter_kw,
    )

    extra_scatters = []
    for j, (featname, feature) in enumerate(extra_features.items()):
        _, scatter = scatter_feature_vs_depth(
            feature,
            depths_um=depths_um,
            amplitudes=amplitudes,
            labels=labels,
            sorting=sorting,
            times_s=times_s,
            motion_est=motion_est,
            registered=registered,
            geom=geom,
            ax=axes.flat[2 + j],
            max_spikes_plot=max_spikes_plot,
            amplitude_color_cutoff=amplitude_color_cutoff,
            amplitude_cmap=amplitude_cmap,
            probe_margin_um=probe_margin_um,
            s=s,
            linewidth=linewidth,
            limits=limits,
            random_seed=random_seed,
            to_show=to_show,
            show_triaged=show_triaged,
            **scatter_kw,
        )
        extra_scatters.append(scatter)
        if label_axes:
            axes.flat[2 + j].set_xlabel(featname)

    _, s_t = scatter_time_vs_depth(
        times_s=times_s,
        depths_um=depths_um,
        amplitudes=amplitudes,
        labels=labels,
        sorting=sorting,
        motion_est=motion_est,
        registered=registered,
        geom=geom,
        probe_margin_um=probe_margin_um,
        ax=axes.flat[-1],
        max_spikes_plot=max_spikes_plot,
        amplitude_color_cutoff=amplitude_color_cutoff,
        amplitude_cmap=amplitude_cmap,
        random_seed=random_seed,
        s=s,
        limits=limits,
        linewidth=linewidth,
        to_show=to_show,
        amplitudes_dataset_name=amplitudes_dataset_name,
        localizations_dataset_name=localizations_dataset_name,
        show_triaged=show_triaged,
        **scatter_kw,
    )

    if label_axes:
        axes[0].set_ylabel(("registered " * registered) + "depth (um)")
        axes[0].set_xlabel("x (um)")
        axes[1].set_xlabel("amplitude (su)")
        axes[-1].set_xlabel("time (s)")

    return figure, axes, (s_x, s_a, *extra_scatters, s_t)


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
    amplitudes_dataset_name="denoised_ptp_amplitudes",
    localizations_dataset_name="point_source_localizations",
    show_triaged=True,
    time_range=None,
    **scatter_kw,
):
    """Scatter plot of spike time vs spike depth (vertical position on probe)

    In this plot, each dot is a spike. Spike data can be passed directly via
    the times_s, depths_um, and (one of) amplitudes or labels as arrays, or
    alternatively, these can be left unset and they will be loaded from
    hdf5_filename when it is supplied.

    Returns: axis, scatter
    """
    if sorting is not None:
        if times_s is None:
            times_s = getattr(sorting, "times_seconds", None)
        if depths_um is None:
            depths_um = getattr(sorting, localizations_dataset_name, None)
            if depths_um is not None:
                depths_um = depths_um[:, 2]
        if amplitudes is None:
            amplitudes = getattr(sorting, amplitudes_dataset_name, None)
        if hdf5_filename is None:
            hdf5_filename = sorting.parent_h5_path

    needs_load = any(v is None for v in (times_s, depths_um, amplitudes, geom))
    if needs_load and hdf5_filename is not None:
        with h5py.File(hdf5_filename, "r", locking=False) as h5:
            if times_s is None:
                times_s = h5["times_seconds"][:]
            if depths_um is None:
                depths_um = h5[localizations_dataset_name][:, 2]
            if amplitudes is None:
                amplitudes = h5[amplitudes_dataset_name][:]
            if geom is None:
                geom = h5["geom"][:]

    return scatter_feature_vs_depth(
        times_s,
        depths_um,
        times_s=times_s,
        amplitudes=amplitudes,
        labels=labels,
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
        limits=limits,
        linewidth=linewidth,
        random_seed=random_seed,
        to_show=to_show,
        show_triaged=show_triaged,
        time_range=time_range,
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
    amplitudes_dataset_name="denoised_ptp_amplitudes",
    localizations_dataset_name="point_source_localizations",
    show_triaged=True,
    **scatter_kw,
):
    """Scatter plot of spike horizontal pos vs spike depth (vertical position on probe)"""

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
        labels=labels,
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
        limits=limits,
        linewidth=linewidth,
        random_seed=random_seed,
        to_show=to_show,
        show_triaged=show_triaged,
        **scatter_kw,
    )
    if show_geom and geom is not None:
        ax.scatter(*geom.T, **geom_scatter_kw)
    if limits == "probe_margin" and geom is not None:
        ax.set_xlim(
            [
                geom[:, 0].min() - probe_margin_um,
                geom[:, 0].max() + probe_margin_um,
            ]
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
    amplitudes_dataset_name="denoised_ptp_amplitudes",
    localizations_dataset_name="point_source_localizations",
    show_triaged=True,
    **scatter_kw,
):
    """Scatter plot of spike amplitude vs spike depth (vertical position on probe)"""
    if sorting is not None:
        if depths_um is None:
            depths_um = getattr(sorting, localizations_dataset_name, None)
            if depths_um is not None:
                depths_um = x[:, 2]
        if amplitudes is None:
            amplitudes = getattr(sorting, amplitudes_dataset_name, None)
        if hdf5_filename is None:
            hdf5_filename = sorting.parent_h5_path

    needs_load = any(v is None for v in (depths_um, amplitudes, geom))
    if needs_load and hdf5_filename is not None:
        with h5py.File(hdf5_filename, "r", locking=False) as h5:
            if depths_um is None:
                depths_um = h5[localizations_dataset_name][:, 2]
            if amplitudes is None:
                amplitudes = h5[amplitudes_dataset_name][:]
            if geom is None:
                geom = h5["geom"][:]

    ax, s = scatter_feature_vs_depth(
        amplitudes,
        depths_um,
        times_s=times_s,
        amplitudes=amplitudes,
        labels=labels,
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
        limits=limits,
        linewidth=linewidth,
        random_seed=random_seed,
        to_show=to_show,
        show_triaged=show_triaged,
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
    time_range=None,
    rasterized=True,
    show_triaged=True,
    show_ellipses=False,
    scat=None,
    ellip=None,
    max_n_labels=None,
    pad_to_max=False,
    **scatter_kw,
):
    assert feature.shape == depths_um.shape
    if amplitudes is not None:
        assert feature.shape == amplitudes.shape
    assert feature.ndim == 1

    updating = scat is not None
    if updating:
        max_spikes_plot = scat.get_offsets().shape[0]
        assert ax is not None

    if ax is None:
        ax = plt.gca()

    # subset spikes according to margin and sampling
    n_spikes = len(feature)
    if to_show is None:
        to_show = np.arange(n_spikes)

    to_show = to_show[np.isfinite(depths_um[to_show])]
    to_show = to_show[np.isfinite(feature[to_show])]
    if geom is not None:
        to_show = to_show[
            (depths_um[to_show] > geom[:, 1].min() - probe_margin_um)
            & (depths_um[to_show] < geom[:, 1].max() + probe_margin_um)
        ]
    if time_range is not None:
        assert times_s is not None
        to_show = to_show[np.clip(times_s[to_show], *time_range) == times_s[to_show]]
    rg = np.random.default_rng(random_seed)
    if len(to_show) > max_spikes_plot:
        to_show = rg.choice(to_show, size=max_spikes_plot, replace=False)
    else:
        to_show = rg.choice(to_show, size=to_show.size, replace=False)

    if registered:
        assert motion_est is not None
        assert times_s is not None
        depths_um = depths_um.copy()
        depths_um[to_show] = motion_est.correct_s(times_s[to_show], depths_um[to_show])

    # order by amplitude so that high amplitude units show up
    if amplitudes is not None:
        to_show = to_show[np.argsort(amplitudes[to_show])]

    if sorting is not None and labels is None:
        if sorting.labels.max() > 0:
            labels = sorting.labels

    if labels is None:
        c = np.clip(amplitudes[to_show], 0, amplitude_color_cutoff)
        c = amplitude_cmap(c / amplitude_color_cutoff)
        order = slice(None)
        show_ellipses = False
    else:
        order = np.concatenate(
            (
                np.flatnonzero(labels[to_show] < 0),
                np.flatnonzero(labels[to_show] >= 0),
            )
        )
        to_show = to_show[order]
        c = glasbey1024[labels[to_show] % len(glasbey1024)]
        if show_triaged:
            triaged = labels[to_show] < 0
            tc = np.clip(amplitudes[to_show[triaged]], 0, amplitude_color_cutoff)
            tc = plt.cm.binary(tc / amplitude_color_cutoff)
            c[triaged] = tc[..., :3]
        else:
            c = c[labels[to_show] >= 0]
            to_show = to_show[labels[to_show] >= 0]

    feature = feature[to_show]
    depths_um = depths_um[to_show]
    if pad_to_max and len(to_show) < max_spikes_plot:
        n_pad = max_spikes_plot - len(to_show)
        feature = np.pad(feature, (0, n_pad), constant_values=np.nan)
        depths_um = np.pad(depths_um, (0, n_pad), constant_values=np.nan)
        c = np.pad(c, [(0, n_pad), (0, 0)], constant_values=0)

    if updating:
        scat.set_offsets(np.c_[feature, depths_um])
        scat.set_facecolors(c)
    else:
        scat = ax.scatter(
            feature,
            depths_um,
            c=c,
            s=s,
            linewidth=linewidth,
            rasterized=rasterized,
            **scatter_kw,
        )

    if show_ellipses:
        ellip = add_ellipses(
            ax,
            labels,
            feature,
            depths_um,
            to_show,
            ellip,
            pad_to_max,
            max_n_labels,
        )

    if limits == "probe_margin" and geom is not None:
        ax.set_ylim(
            [
                geom[:, 1].min() - probe_margin_um,
                geom[:, 1].max() + probe_margin_um,
            ]
        )
    elif limits is not None and limits != "probe_margin":
        ax.set_ylim(limits)

    if show_ellipses:
        return ax, scat, ellip
    return ax, scat


def add_ellipses(
    ax, labels, feature, depths_um, to_show, ellip, pad_to_max, max_n_labels
):
    unit_ids = np.unique(labels[to_show])
    unit_ids = unit_ids[unit_ids >= 0]
    if pad_to_max and max_n_labels:
        n_pad = max_n_labels - unit_ids.size
        unit_ids = np.concatenate(
            (unit_ids, unit_ids.max() + 10 + np.arange(n_pad))
        )
    if ellip is None:
        ellip = {}
    for uid in unit_ids:
        color = glasbey1024[uid % len(glasbey1024)]
        in_unit = np.flatnonzero(labels[to_show] == uid)
        bad = in_unit.size <= 2
        if not bad:
            # nans
            f = feature[in_unit]
            d = depths_um[in_unit]
            valid = np.flatnonzero(np.isfinite(f + d))
            bad = valid.size <= 2
        if not bad:
            # remove outliers to stabilize [co]variance
            kdt = KDTree(np.c_[f[valid], d[valid]])
            dd, ii = kdt.query(
                np.c_[f[valid], d[valid]], distance_upper_bound=10.0
            )
            valid = valid[ii < kdt.n]
            bad = valid.size <= 2
        if not bad:
            fm = f[valid].mean()
            dm = d[valid].mean()
            cov = np.cov(f[valid], d[valid])
            vx, vy = cov[0, 0], cov[1, 1]
            if min(vx, vy) <= 0:
                bad = True
            rho = np.minimum(1.0, cov[0, 1] / np.sqrt(vx * vy))
            rhoss = cov[0, 1]
        if not bad:
            apc2 = (vx + vy) / 2
            amc2sq = ((vx - vy) ** 2) / 4
            lambda1 = apc2 + np.sqrt(amc2sq + rhoss**2)
            lambda2 = apc2 - np.sqrt(amc2sq + rhoss**2)
            if rhoss == 0:
                theta = (np.pi / 2) * (vx < vy)
            else:
                theta = np.arctan2(lambda1 - vx, rhoss)
            theta = 180 * theta / np.pi
            center = fm, dm
        else:
            center = 0, 0
            color = (0, 0, 0, 0)
            lambda1 = lambda2 = 1
            theta = 0
        if uid in ellip:
            ell = ellip[uid]
            # ell.remove()
            ell.set(
                center=center,
                width=2 * np.sqrt(lambda1),
                height=2 * np.sqrt(lambda2),
                angle=theta,
                edgecolor=color,
            )
        else:
            ell = Ellipse(
                center,
                width=2 * np.sqrt(lambda1),
                height=2 * np.sqrt(lambda2),
                angle=theta,
                facecolor=(0, 0, 0, 0),
                edgecolor=color,
                linewidth=1,
                animated=True,
            )
            ellip[uid] = ell
            ax.add_patch(ell)
    return ellip
