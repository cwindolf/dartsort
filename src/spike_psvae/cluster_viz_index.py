from pathlib import Path

import colorcet as cc
import h5py
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.patches import Ellipse, Rectangle
from matplotlib_venn import venn2
from scipy.spatial.distance import cdist
from spikeinterface import NumpySorting
from spikeinterface.comparison import compare_two_sorters
from spikeinterface.postprocessing import compute_correlograms
from tqdm.auto import tqdm

from . import chunk_features, spikeio, waveform_utils
from .pyks_ccg import ccg

ccolors = cc.glasbey[:31]


def get_ccolor(k):
    if k == -1:
        return "#808080"
    else:
        return ccolors[k % len(ccolors)]


def cluster_scatter(
    xs,
    ys,
    ids,
    ax=None,
    n_std=2.0,
    excluded_ids={-1},
    s=1,
    alpha=0.5,
    annotate=True,
    do_ellipse=True,
):
    ax = ax or plt.gca()
    # scatter and collect gaussian info
    means = {}
    covs = {}
    for k in np.unique(ids):
        where = np.flatnonzero(ids == k)
        xk = xs[where]
        yk = ys[where]
        color = get_ccolor(k)
        ax.scatter(
            xk, yk, s=s, color=color, alpha=alpha, marker=".", rasterized=True
        )
        if k not in excluded_ids:
            if do_ellipse:
                x_mean, y_mean = xk.mean(), yk.mean()
                means[k] = x_mean, y_mean
                if where.size > 2:
                    xycov = np.cov(xk, yk)
                    covs[k] = xycov
                else:
                    covs[k] = np.zeros((2, 2))
            if annotate:
                ax.annotate(str(k), (x_mean, y_mean), size=6)

    if not do_ellipse:
        return

    for k in means.keys():
        if (ids == k).sum() > 0:
            mean_x, mean_y = means[k]
            cov = covs[k]

            vx, vy = cov[0, 0], cov[1, 1]
            if min(vx, vy) <= 0:
                continue
            rho = np.minimum(1.0, cov[0, 1] / np.sqrt(vx * vy))

            color = get_ccolor(k)
            ell = Ellipse(
                (0, 0),
                width=2 * np.sqrt(1 + rho),
                height=2 * np.sqrt(1 - rho),
                facecolor=(0, 0, 0, 0),
                edgecolor=color,
                linewidth=1,
            )
            transform = (
                transforms.Affine2D()
                .rotate_deg(45)
                .scale(n_std * np.sqrt(vx), n_std * np.sqrt(vy))
                .translate(mean_x, mean_y)
            )
            ell.set_transform(transform + ax.transData)
            ax.add_patch(ell)


def array_scatter(
    labels,
    geom,
    x,
    z,
    maxptp,
    zlim=None,
    axes=None,
    annotate=True,
    subplots_kw={},
    do_ellipse=True,
    c=0,
    do_log=True,
):
    fig = None
    if axes is None:
        subkw = dict(sharey=True, figsize=(15, 15))
        subkw.update(subplots_kw)
        fig, axes = plt.subplots(1, 3, **subkw)

    cluster_scatter(
        x,
        z,
        labels,
        ax=axes[0],
        s=10,
        alpha=0.05,
        annotate=annotate,
        do_ellipse=do_ellipse,
    )
    if geom is not None:
        axes[0].scatter(*geom.T, c="orange", marker="s", s=10)
    axes[0].set_ylabel("z")
    axes[0].set_xlabel("x")
    axes[2].set_xlabel("x")

    cluster_scatter(
        np.log(c + maxptp) if do_log else maxptp,
        z,
        labels,
        ax=axes[1],
        s=10,
        alpha=0.05,
        annotate=annotate,
        do_ellipse=do_ellipse,
    )
    xlabel = "maxptp"
    if do_log and c > 0:
        xlabel = f"log({c}+maxptp)"
    elif do_log:
        xlabel = "log maxptp"
    axes[1].set_xlabel(xlabel)
    axes[2].scatter(
        x,
        z,
        c=np.clip(maxptp, 3, 15),
        alpha=0.05,
        s=10,
        marker=".",
        cmap=plt.cm.viridis,
    )
    if geom is not None:
        axes[2].scatter(*geom.T, c="orange", marker="s", s=10)
    if zlim is not None:
        axes[0].set_ylim(zlim)
        axes[2].set_ylim(zlim)
        axes[1].set_ylim(zlim)

    if fig is not None:
        plt.tight_layout()

    return fig, axes


def pgeom(
    waveforms,
    max_channels=None,
    channel_index=None,
    geom=None,
    ax=None,
    color=None,
    alpha=1,
    z_extension=1.0,
    x_extension=0.8,
    lw=None,
    ls=None,
    show_zero=True,
    show_zero_kwargs=None,
    max_abs_amp=None,
    show_chan_label=False,
    chan_labels=None,
    linestyle=None,
    xlim_factor=1,
    subar=False,
    msbar=False,
    zlim="tight",
    rasterized=False,
):
    """Plot waveforms according to geometry using channel index"""
    assert geom is not None
    ax = ax or plt.gca()

    # -- validate shapes
    if max_channels is None and channel_index is None:
        max_channels = np.zeros(waveforms.shape[0], dtype=int)
        channel_index = (
            np.arange(geom.shape[0])[None, :]
            * np.ones(geom.shape[0], dtype=int)[:, None]
        )
    max_channels = np.atleast_1d(max_channels)
    if waveforms.ndim == 2:
        waveforms = waveforms[None]
    else:
        assert waveforms.ndim == 3
    n_channels, C = channel_index.shape
    assert geom.shape == (n_channels, 2)
    T = waveforms.shape[1]
    if waveforms.shape != (*max_channels.shape, T, C):
        raise ValueError(
            f"Bad shapes: {waveforms.shape=}, {max_channels.shape=}"
        )

    # -- figure out units for plotting
    z_uniq, z_ix = np.unique(geom[:, 1], return_inverse=True)
    for i in z_ix:
        x_uniq = np.unique(geom[z_ix == i, 0])
        if x_uniq.size > 1:
            break
    else:
        x_uniq = np.unique(geom[:, 0])
    inter_chan_x = 1
    if x_uniq.size > 1:
        inter_chan_x = x_uniq[1] - x_uniq[0]
    inter_chan_z = z_uniq[1] - z_uniq[0]
    max_abs_amp = max_abs_amp or np.nanmax(np.abs(waveforms))
    geom_scales = [
        T / inter_chan_x / x_extension,
        max_abs_amp / inter_chan_z / z_extension,
    ]
    geom_plot = geom * geom_scales
    t_domain = np.linspace(-T // 2, T // 2, num=T)

    if subar:
        if isinstance(subar, bool):
            # auto su if True, but you can pass int/float too.
            subars = (1, 2, 5, 10, 20, 30, 50)
            for j in range(len(subars)):
                if subars[j] > 1.2 * max_abs_amp:
                    break
            subar = subars[max(0, j - 1)]

    # -- and, plot
    draw = []
    unique_chans = set()
    xmin, xmax = np.inf, -np.inf
    for wf, mc in zip(waveforms, max_channels):
        for i, c in enumerate(channel_index[mc]):
            if c == n_channels:
                continue

            draw.append(geom_plot[c, 0] + t_domain)
            draw.append(geom_plot[c, 1] + wf[:, i])
            xmin = min(geom_plot[c, 0] + t_domain.min(), xmin)
            xmax = max(geom_plot[c, 0] + t_domain.max(), xmax)
            unique_chans.add(c)
    dx = xmax - xmin
    ax.set_xlim(
        [
            xmin + dx / 2 - xlim_factor * dx / 2,
            xmax - dx / 2 + xlim_factor * dx / 2,
        ]
    )

    ann_offset = np.array([0, 0.33 * inter_chan_z]) * geom_scales
    chan_labels = (
        chan_labels
        if chan_labels is not None
        else list(map(str, range(len(geom))))
    )
    for c in unique_chans:
        if show_zero:
            if show_zero_kwargs is None:
                show_zero_kwargs = dict(color="gray", lw=1, linestyle="--")
            ax.axhline(geom_plot[c, 1], **show_zero_kwargs)
        if show_chan_label:
            ax.annotate(
                chan_labels[c], geom_plot[c] + ann_offset, size=6, color="gray"
            )
    linestyle = linestyle or ls
    lines = ax.plot(
        *draw,
        alpha=alpha,
        color=color,
        lw=lw,
        linestyle=linestyle,
        rasterized=rasterized,
    )

    if subar:
        min_z = min(geom_plot[c, 1] for c in unique_chans)
        if msbar:
            min_z += max_abs_amp
        ax.add_patch(
            Rectangle(
                [
                    geom_plot[:, 0].max() + T // 4,
                    min_z - max_abs_amp / 2,
                ],
                4,
                subar,
                fc="k",
            )
        )
        ax.text(
            geom_plot[:, 0].max() + T // 4 + 4 + 5,
            min_z - max_abs_amp / 2 + subar / 2,
            f"{subar} s.u.",
            transform=ax.transData,
            fontsize=5,
            ha="left",
            va="center",
            rotation=-90,
        )

    if msbar:
        min_z = min(geom_plot[c, 1] for c in unique_chans)
        ax.plot(
            [
                geom_plot[:, 0].max() - 30,
                geom_plot[:, 0].max(),
            ],
            2 * [min_z - max_abs_amp],
            color="k",
            lw=2,
            zorder=890,
        )
        ax.text(
            geom_plot[:, 0].max() - 15,
            min_z - max_abs_amp + max_abs_amp / 10,
            "1ms",
            transform=ax.transData,
            fontsize=5,
            ha="center",
            va="bottom",
        )

    if zlim is None:
        pass
    elif zlim == "auto":
        min_z = min(geom_plot[c, 1] for c in unique_chans)
        max_z = max(geom_plot[c, 1] for c in unique_chans)
        if np.isfinite([min_z, max_z]).all():
            ax.set_ylim([min_z - 2 * max_abs_amp, max_z + 2 * max_abs_amp])
    elif zlim == "tight":
        min_z = min(geom_plot[c, 1] for c in unique_chans)
        max_z = max(geom_plot[c, 1] for c in unique_chans)
        if np.isfinite([min_z, max_z]).all():
            ax.set_ylim([min_z - max_abs_amp, max_z + max_abs_amp])
    elif isinstance(zlim, float):
        min_z = min(geom_plot[c, 1] for c in unique_chans)
        max_z = max(geom_plot[c, 1] for c in unique_chans)
        if np.isfinite([min_z, max_z]).all():
            ax.set_ylim(
                [min_z - max_abs_amp * zlim, max_z + max_abs_amp * zlim]
            )

    return lines


def superres_template_viz(
    orig_label,
    superres_templates,
    superres_label_to_orig_label,
    superres_label_to_bin_id,
    spike_train_orig,
    geom,
    radius=200,
):
    in_label = np.flatnonzero(superres_label_to_orig_label == orig_label)
    max_bin = np.abs(superres_label_to_bin_id).max()
    temps = superres_templates[in_label]
    bin_ids = superres_label_to_bin_id[in_label]
    colors = 0.5 + bin_ids / max_bin / 2
    ns = (spike_train_orig[:, 1] == orig_label).sum()

    channel_index = waveform_utils.make_channel_index(geom, radius)
    avg_temp = temps.mean(0)
    my_mc = avg_temp.ptp(0).argmax()

    fig, ax = plt.subplots(figsize=(6, 6))
    for temp, col in zip(temps, colors):
        temp_pad = np.pad(temp, [(0, 0), (0, 1)], constant_values=np.nan)
        pgeom(
            temp_pad[:, channel_index[my_mc]],
            my_mc,
            channel_index,
            geom,
            ax=ax,
            color=plt.cm.viridis(col),
            max_abs_amp=np.abs(temps).max(),
            subar=True,
            show_chan_label=True,
        )
    ax.axis("off")
    ax.set_title(
        f"Superres templates for unit={orig_label}. {ns} spikes, {temps.ptp(1).max():0.1f} max ptp."
    )

    return fig, ax


def superres_templates_viz(
    superres_templates,
    superres_label_to_orig_label,
    superres_label_to_bin_id,
    spike_train_orig,
    output_directory,
    geom,
    radius=200,
):
    output_directory.mkdir(exist_ok=True, parents=True)
    for orig_label in tqdm(np.unique(superres_label_to_orig_label)):
        fig, ax = superres_template_viz(
            orig_label,
            superres_templates,
            superres_label_to_orig_label,
            superres_label_to_bin_id,
            spike_train_orig,
            geom,
            radius=radius,
        )
        fig.savefig(
            output_directory / f"superres_unit{orig_label:03d}.png", dpi=300
        )
        plt.close(fig)


def reassignment_viz(
    orig_label,
    spike_train_orig,
    new_labels,
    raw_bin,
    geom,
    templates=None,
    radius=200,
    n_plot=250,
    z_extension=1.0,
    trough_offset=42,
    spike_length_samples=121,
    proposed_pairs=None,
    reassigned_scores=None,
    reassigned_resids=None,
    reas_channel_index=None,
    max_channels=None,
):
    in_unit = np.flatnonzero(spike_train_orig[:, 1] == orig_label)
    newids = new_labels[in_unit]
    new_units = np.setdiff1d(np.unique(newids), [orig_label])
    kept = newids == orig_label
    # print(orig_label, new_units, np.unique(newids))
    # print(in_unit.size, in_unit[kept].size)

    show_scores = reassigned_scores is not None
    show_resids = reassigned_resids is not None
    fig, axes = plt.subplots(
        nrows=1 + show_scores + show_resids,
        ncols=1 + new_units.size,
        figsize=(
            8 * (1 + new_units.size),
            8 + 4 * show_scores + 8 * show_resids,
        ),
        gridspec_kw=dict(
            height_ratios=[2] + ([1] * show_scores) + ([2] * show_resids),
        ),
        squeeze=False,
    )
    axes = np.atleast_1d(axes)

    rg = np.random.default_rng(0)
    channel_index = waveform_utils.make_channel_index(geom, radius)

    orig_choices = rg.choice(
        in_unit, size=min(n_plot, in_unit.size), replace=False
    )
    orig_choices.sort()
    orig_wf, skipped = spikeio.read_waveforms(
        spike_train_orig[orig_choices, 0],
        raw_bin,
        geom.shape[0],
        channel_index=None,
        max_channels=None,
        channels=None,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )
    orig_choices = np.delete(orig_choices, skipped)
    og_temp = orig_wf.mean(0)
    og_mc = og_temp.ptp(0).argmax()
    orig_wf = np.pad(orig_wf, [(0, 0), (0, 0), (0, 1)])[
        :, :, channel_index[og_mc]
    ]
    og_mcs = np.array([og_mc] * orig_wf.shape[0])

    kept_choices = rg.choice(
        in_unit[kept], size=min(n_plot, in_unit[kept].size), replace=False
    )
    kept_choices.sort()
    kept_wf, skipped = spikeio.read_waveforms(
        spike_train_orig[kept_choices, 0],
        raw_bin,
        geom.shape[0],
        channels=channel_index[og_mc],
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )
    kept_choices = np.delete(kept_choices, skipped)
    kept_mcs = np.array([og_mc] * kept_wf.shape[0])
    pgeom(
        orig_wf,
        og_mcs,
        channel_index,
        geom,
        ax=axes[0, 0],
        color="k",
        max_abs_amp=np.abs(orig_wf).max(),
        lw=1,
        alpha=0.1,
        z_extension=z_extension,
        show_zero=False,
    )
    if kept_choices.size:
        pgeom(
            kept_wf,
            kept_mcs,
            channel_index,
            geom,
            ax=axes[0, 0],
            color=cc.glasbey[0],
            max_abs_amp=np.abs(orig_wf).max(),
            lw=1,
            alpha=0.1,
            show_zero=False,
            z_extension=z_extension,
        )
    pairstr = ""
    if proposed_pairs is not None:
        pairstr = (
            f" proppairs: {','.join(map(str, proposed_pairs[orig_label]))}"
        )
    axes[0, 0].set_title(
        f"unit {orig_label} kept {kept.sum()}/{in_unit.size} ({100*kept.mean():0.1f}%){pairstr}"
    )

    newchoices = []
    for j, (newu, ax) in enumerate(zip(new_units, axes[0])):
        new_choices = rg.choice(
            in_unit[newids == newu],
            size=min(n_plot, in_unit[newids == newu].size),
            replace=False,
        )
        new_choices.sort()
        axes[0, j + 1].set_title(f"{(newids == newu).sum()} spikes -> {newu}")
        new_wf, skipped = spikeio.read_waveforms(
            spike_train_orig[new_choices, 0],
            raw_bin,
            geom.shape[0],
            channels=channel_index[og_mc],
            trough_offset=trough_offset,
            spike_length_samples=spike_length_samples,
        )
        newchoices.append(np.delete(new_choices, skipped))
        new_mcs = [og_mc] * new_wf.shape[0]
        pgeom(
            orig_wf,
            og_mcs,
            channel_index,
            geom,
            ax=axes[0, j + 1],
            color="k",
            lw=1,
            alpha=0.1,
            z_extension=z_extension,
        )
        if new_choices.size:
            pgeom(
                new_wf,
                new_mcs,
                channel_index,
                geom,
                ax=axes[0, j + 1],
                color=cc.glasbey[j + 1],
                max_abs_amp=np.abs(orig_wf).max(),
                lw=1,
                alpha=0.5,
                show_zero=False,
                z_extension=z_extension,
            )

    # plot templates on top
    if templates is not None:
        templates = np.pad(
            templates, [(0, 0), (0, 0), (0, 1)], constant_values=np.nan
        )
        for ax in axes[0]:
            for c, ls in zip((cc.glasbey[0], "w"), (None, ":")):
                pgeom(
                    templates[orig_label][None, :, channel_index[og_mc]],
                    [og_mc],
                    channel_index,
                    geom,
                    color=c,
                    max_abs_amp=np.abs(orig_wf).max(),
                    lw=1,
                    ls=ls,
                    alpha=1,
                    show_zero=False,
                    z_extension=z_extension,
                    ax=ax,
                )
    for j, newu in enumerate(new_units):
        if templates is not None:
            # for ax in axes[0]:
            for c, ls in zip((cc.glasbey[j + 1], "w"), (None, ":")):
                pgeom(
                    templates[newu][None, :, channel_index[og_mc]],
                    [og_mc],
                    channel_index,
                    geom,
                    color=c,
                    max_abs_amp=np.abs(orig_wf).max(),
                    lw=1,
                    ls=ls,
                    alpha=1,
                    show_zero=False,
                    z_extension=z_extension,
                    ax=axes[0, j + 1],
                )

    # plot outlier scores
    if show_scores:
        axes[1, 0].scatter(
            *reassigned_scores[in_unit].T, color="k", s=2, lw=0, label="all"
        )
        axes[1, 0].scatter(
            *reassigned_scores[in_unit[kept]].T,
            color=cc.glasbey[0],
            s=2,
            lw=0,
            label="kept",
        )

        for j, (newu, ax) in enumerate(zip(new_units, axes[1, 1:])):
            ax.scatter(
                *reassigned_scores[in_unit].T,
                color="k",
                s=2,
                lw=0,
                label="all",
            )
            ax.scatter(
                *reassigned_scores[in_unit[newids == newu]].T,
                color=cc.glasbey[j + 1],
                s=2,
                lw=0,
                label=f"to {newu}",
            )

        for ax in axes[1]:
            ax.set_xlabel("original unit outlier score")
            ax.set_ylabel("reassigned unit outlier score")
            mn = reassigned_scores[in_unit].min(axis=0)
            mx = reassigned_scores[in_unit].max(axis=0)
            ax.plot(
                [max(mn[0], mn[1]), min(mx[0], mx[1])],
                [max(mn[0], mn[1]), min(mx[0], mx[1])],
                color="gray",
                lw=1,
                label="y=x",
            )
            ax.legend(frameon=False, loc="upper left")

    # plot residuals
    if show_resids:
        assert reas_channel_index is not None
        assert max_channels is not None
        orig_resids = waveform_utils.restrict_wfs_to_chans(
            np.stack([reassigned_resids[j, 0] for j in orig_choices]),
            max_channels=max_channels[orig_choices],
            channel_index=reas_channel_index,
            dest_channels=channel_index[og_mc],
        )
        pgeom(
            orig_resids,
            og_mcs,
            channel_index,
            geom,
            ax=axes[2, 0],
            color="k",
            max_abs_amp=np.nanmax(np.abs(orig_resids)),
            lw=1,
            alpha=0.1,
            z_extension=z_extension,
            show_zero=False,
        )
        if kept_choices.size:
            kept_resids = waveform_utils.restrict_wfs_to_chans(
                np.stack([reassigned_resids[j, 1] for j in kept_choices]),
                max_channels=max_channels[kept_choices],
                channel_index=reas_channel_index,
                dest_channels=channel_index[og_mc],
            )
            pgeom(
                kept_resids,
                kept_mcs,
                channel_index,
                geom,
                ax=axes[2, 0],
                color=cc.glasbey[0],
                max_abs_amp=np.nanmax(np.abs(orig_resids)),
                lw=1,
                alpha=0.1,
                z_extension=z_extension,
                show_zero=False,
            )

        for j, (newu, ax) in enumerate(zip(new_units, axes[2, 1:])):
            newu_resids_orig = waveform_utils.restrict_wfs_to_chans(
                np.stack([reassigned_resids[jj, 0] for jj in newchoices[j]]),
                max_channels=max_channels[newchoices[j]],
                channel_index=reas_channel_index,
                dest_channels=channel_index[og_mc],
            )
            pgeom(
                newu_resids_orig,
                [og_mc] * newu_resids_orig.shape[0],
                channel_index,
                geom,
                ax=ax,
                color="k",
                max_abs_amp=np.nanmax(np.abs(orig_resids)),
                lw=1,
                alpha=0.1,
                z_extension=z_extension,
                show_zero=False,
            )

            newu_resids_new = waveform_utils.restrict_wfs_to_chans(
                np.stack([reassigned_resids[jj, 1] for jj in newchoices[j]]),
                max_channels=max_channels[newchoices[j]],
                channel_index=reas_channel_index,
                dest_channels=channel_index[og_mc],
            )
            pgeom(
                newu_resids_new,
                [og_mc] * newu_resids_new.shape[0],
                channel_index,
                geom,
                ax=ax,
                color=cc.glasbey[j + 1],
                max_abs_amp=np.nanmax(np.abs(orig_resids)),
                lw=1,
                alpha=0.1,
                z_extension=z_extension,
                show_zero=False,
            )

    return fig, axes


def reassignments_viz(
    spike_train_orig,
    new_labels,
    raw_bin,
    output_directory,
    geom,
    units=None,
    templates=None,
    radius=200,
    z_extension=1.0,
    trough_offset=42,
    spike_length_samples=121,
    proposed_pairs=None,
    reassigned_scores=None,
    reassigned_resids=None,
    reas_channel_index=None,
    max_channels=None,
):
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)
    if units is None:
        units = np.arange(spike_train_orig[:, 1].max() + 1)
    units = np.intersect1d(
        units, np.setdiff1d(np.unique(spike_train_orig[:, 1]), [-1])
    )
    for orig_label in tqdm(units):
        fig, ax = reassignment_viz(
            orig_label,
            spike_train_orig,
            new_labels,
            raw_bin,
            geom,
            templates=templates,
            radius=radius,
            z_extension=z_extension,
            trough_offset=trough_offset,
            spike_length_samples=spike_length_samples,
            proposed_pairs=proposed_pairs,
            reassigned_scores=reassigned_scores,
            reassigned_resids=reassigned_resids,
            reas_channel_index=reas_channel_index,
            max_channels=max_channels,
        )
        fig.savefig(
            output_directory / f"reassign_unit{orig_label:03d}.png", dpi=300
        )
        plt.close(fig)


def unsorted_waveforms_diagnostic(
    subtraction_h5, raw_bin, nwfs=50, cutoff=15, trough_offset=42
):
    rg = np.random.default_rng(0)

    with h5py.File(subtraction_h5) as h5:
        maxptps = h5["maxptps"][:]
        which_high_amp = np.flatnonzero(maxptps > cutoff)
        which_high_amp = rg.choice(
            which_high_amp, size=min(nwfs, which_high_amp.size), replace=False
        )
        which_low_amp = np.flatnonzero(maxptps < cutoff)
        which_low_amp = rg.choice(
            which_low_amp, size=min(nwfs, which_low_amp.size), replace=False
        )

        # load denoised waveforms
        tpca = chunk_features.TPCA(which_waveforms="denoised")
        tpca.from_h5(h5)

        # load raw waveforms
        raw_high = spikeio.read_waveforms()

    raise NotImplementedError


def plot_waveforms_geom(
    main_cluster_id,
    neighbor_clusters,
    labels,
    geom,
    channel_index,
    spike_index,
    maxptps,
    all_waveforms=None,
    raw_bin=None,
    residual_bin=None,
    spikes_plot=100,
    t_range=(22, 72),
    num_rows=3,
    alpha=0.1,
    h_shift=0,
    do_mean=False,
    ax=None,
    colors=None,
    scale_mul=1,
):
    ax = ax or plt.gca()

    # what channels will we plot?
    vals, counts = np.unique(
        spike_index[np.flatnonzero(labels == main_cluster_id), 1],
        return_counts=True,
    )
    z_uniq, z_ids = np.unique(geom[:, 1], return_inverse=True)
    mcid = z_ids[vals[counts.argmax()]]
    channels_plot = np.flatnonzero((z_ids >= mcid - 3) & (z_ids <= mcid + 3))

    # how to scale things?
    all_max_ptp = maxptps[
        np.isin(labels, (*neighbor_clusters, main_cluster_id))
    ].max()
    scale = scale_mul * (z_uniq[1] - z_uniq[0]) / max(7, all_max_ptp)

    times_plot = np.arange(t_range[0] - 42, t_range[1] - 42).astype(float)
    x_uniq = np.unique(geom[:, 0])
    times_plot *= (x_uniq[1] - x_uniq[0]) / np.abs(times_plot).max()

    # scatter the channels
    ax.scatter(*geom[channels_plot].T, c="orange", marker="s")
    for c in channels_plot:
        ax.annotate(c, (geom[c, 0], geom[c, 1]))

    if raw_bin:
        raw_data = np.memmap(raw_bin, dtype=np.float32)
        raw_data = raw_data.reshape(-1, len(channel_index))

    if residual_bin:
        res_data = np.memmap(residual_bin, dtype=np.float32)
        res_data = res_data.reshape(-1, len(channel_index))

    for j, cluster_id in reversed(
        list(enumerate((main_cluster_id, *neighbor_clusters)))
    ):
        if colors is None:
            color = get_ccolor(cluster_id)
        else:
            color = colors[j]
        in_cluster = np.flatnonzero(labels == cluster_id)
        num_plot_cluster = min(len(in_cluster), spikes_plot)
        some_in_cluster = np.random.default_rng(0).choice(
            in_cluster, replace=False, size=num_plot_cluster
        )
        some_in_cluster.sort()

        if raw_bin:
            spike_times = spike_index[some_in_cluster][:, 0]
            waveforms = []
            for t in spike_times:
                waveforms.append(
                    raw_data[t - 42 : t + 79, channels_plot].copy()
                )
            waveforms = np.asarray(waveforms)
        else:
            waveforms = all_waveforms[some_in_cluster]

        if residual_bin:
            spike_times = spike_index[some_in_cluster][:, 0]
            residuals = []
            for t in spike_times:
                residuals.append(
                    res_data[t - 42 : t + 79, channels_plot].copy()
                )
            residuals = np.asarray(residuals)
        if do_mean:
            waveforms = np.expand_dims(np.mean(waveforms, axis=0), 0)

        vertical_lines = set()
        draw_lines = []
        for i in range(num_plot_cluster):
            if raw_bin:
                wf_chans = channels_plot
            else:
                wf_chans = channel_index[spike_index[some_in_cluster[i], 1]]

            for k, channel in enumerate(channels_plot):
                if channel in wf_chans:
                    trace = waveforms[
                        i,
                        t_range[0] : t_range[1],
                        np.flatnonzero(np.isin(wf_chans, channel))[0],
                    ]
                else:
                    continue
                if residual_bin:
                    trace += residuals[i, t_range[0] : t_range[1], k]

                waveform = trace * scale
                draw_lines.append(geom[channel, 0] + times_plot)
                draw_lines.append(waveform + geom[channel, 1])
                max_vert_line = geom[channel, 0]
                if max_vert_line not in vertical_lines:
                    vertical_lines.add(max_vert_line)
                    ax.axvline(max_vert_line, linestyle="--")

        ax.plot(
            *draw_lines,
            alpha=alpha,
            c=color,
        )


def plot_ccg(times, nbins=50, ms_frames=30, ax=None):
    ax = ax or plt.gca()
    nbins = 50
    ccg_ = ccg(times, times, nbins, ms_frames)
    ccg_[nbins] = 0

    ax.bar(np.arange(nbins * 2 + 1) - nbins, ccg_, width=1, ec="none")
    ax.set_xlabel("isi (ms)")
    ax.set_ylabel("autocorrelogram count")


single_unit_mosaic = """\
abc.pppqqqqrrr
abc.pppqqqqrrr
abc.xxxxxyyyyy
abc.ii..jj..kk
"""


def single_unit_summary(
    cluster_id,
    clusterer,
    labels,
    geom,
    kept_inds,
    x,
    z,
    maxptps,
    channel_index,
    spike_index,
    wfs_localized,
    wfs_subtracted,
    raw_bin,
    residual_bin,
    spikes_plot=100,
    num_rows=3,
    alpha=0.1,
    scale_mul=1,
):
    # 2 neighbor clusters
    cluster_centers = np.array(
        [
            clusterer.weighted_cluster_centroid(l)
            for l in np.setdiff1d(np.unique(labels), [-1])
        ]
    )
    closest_clusters = np.argsort(
        cdist([cluster_centers[cluster_id]], cluster_centers)[0]
    )[1:3]

    fig, axes = plt.subplot_mosaic(
        single_unit_mosaic,
        figsize=(15, 10),
        gridspec_kw=dict(
            hspace=0.5,
            wspace=0.1,
            width_ratios=[
                4,
                4,
                4,
                1,
                1,
                1,
                0.33,
                0.167,
                1,
                1,
                0.167,
                0.33,
                1,
                1,
            ],
        ),
    )
    axes["a"].get_shared_y_axes().join(axes["a"], axes["b"])
    axes["a"].get_shared_y_axes().join(axes["a"], axes["c"])
    axes["p"].get_shared_y_axes().join(axes["p"], axes["q"])
    axes["p"].get_shared_y_axes().join(axes["p"], axes["r"])
    axes["x"].get_shared_y_axes().join(axes["x"], axes["y"])

    # -- waveform plots
    for ax, w, raw, res, title in zip(
        "abc",
        [None, wfs_subtracted, wfs_localized],
        [raw_bin, None, None],
        [None, residual_bin, None],
        ["raw", "cleaned", "denoised"],
    ):
        plot_waveforms_geom(
            cluster_id,
            closest_clusters,
            labels,
            geom,
            channel_index,
            spike_index,
            maxptps,
            all_waveforms=w,
            raw_bin=raw,
            residual_bin=res,
            spikes_plot=spikes_plot,
            num_rows=num_rows,
            alpha=alpha,
            ax=axes[ax],
            scale_mul=scale_mul,
        )
        axes[ax].set_title(title)
    axes["b"].set_yticks([])
    axes["c"].set_yticks([])

    # -- scatter plots
    in_shown_clusters = np.flatnonzero(
        np.isin(labels, (*closest_clusters, cluster_id))
    )
    zlim = (z[in_shown_clusters].min(), z[in_shown_clusters].max())
    array_scatter(
        labels[in_shown_clusters],
        geom,
        x[in_shown_clusters],
        z[in_shown_clusters],
        maxptps[in_shown_clusters],
        zlim=zlim,
        axes=[axes["p"], axes["q"], axes["r"]],
    )
    axes["q"].set_yticks([])
    axes["r"].set_yticks([])

    # -- this unit stats
    in_main_cluster = np.flatnonzero(labels == cluster_id)
    s_cluster = spike_index[in_main_cluster, 0]
    t_cluster = s_cluster / 30000
    maxptp_cluster = maxptps[in_main_cluster]
    z_cluster = z[in_main_cluster]

    # ptp vs t plot
    axes["x"].plot(t_cluster, maxptp_cluster)
    axes["x"].set_xlabel("t (s)")
    axes["x"].set_ylabel("ptp")

    # ptp vs z plot
    axes["y"].scatter(z_cluster, maxptp_cluster)
    axes["y"].set_xlabel("z")
    # axes["y"].set_ylabel("ptp")
    axes["y"].set_yticks([])

    # ISI plot
    isi_ms = 1000 * np.diff(t_cluster)
    axes["k"].hist(isi_ms, bins=np.arange(11))
    axes["k"].set_xlabel("isi (ms)")
    axes["k"].set_ylabel("count")

    # cross correlograms
    for ax, unit in zip("ij", closest_clusters):
        in_other = np.flatnonzero(labels == unit)
        sorting = NumpySorting.from_times_labels(
            times_list=np.r_[s_cluster, spike_index[in_other, 0]],
            labels_list=np.r_[
                np.zeros(len(s_cluster), dtype=int),
                np.ones(len(in_other), dtype=int),
            ],
            sampling_frequency=30000,
        )
        correlograms, bins = compute_correlograms(
            sorting, symmetrize=True, window_ms=10.0, bin_ms=1.0
        )
        axes[ax].bar(bins[1:], correlograms[0][1], width=1.0, align="center")
        axes[ax].set_xticks([bins[0], 0, bins[-1]])
        axes[ax].set_xlabel("lag (ms)")
        axes[ax].set_title(f"ccg{cluster_id} <-> {unit}")

    return fig


def plot_agreement_venn(
    cluster_id1,
    geom,
    channel_index,
    spike_index1,
    spike_index2,
    labels1,
    labels2,
    name1,
    name2,
    raw_bin,
    maxptps1,
    maxptps2,
    match_score=0.5,
    spikes_plot=100,
    delta_frames=12,
):
    # make spikeinterface objects
    sorting1 = NumpySorting.from_times_labels(
        times_list=spike_index1[:, 0],
        labels_list=labels1,
        sampling_frequency=30000,
    )
    sorting2 = NumpySorting.from_times_labels(
        times_list=spike_index2[:, 0],
        labels_list=labels2,
        sampling_frequency=30000,
    )
    comp = compare_two_sorters(
        sorting1,
        sorting2,
        sorting1_name=name1,
        sorting2_name=name2,
        match_score=0.5,
    )

    # get best match
    match2 = comp.get_best_unit_match1(cluster_id1)
    if not (match2 and match2 > -1):
        return
    match_frac = comp.get_agreement_fraction(cluster_id1, match2)

    st1 = sorting1.get_unit_spike_train(cluster_id1)
    st2 = sorting2.get_unit_spike_train(match2)
    (
        ind_st1,
        ind_st2,
        not_match_ind_st1,
        not_match_ind_st2,
    ) = compute_spiketrain_agreement(
        st1,
        st2,
        delta_frames,
    )
    fig, axes = plt.subplots(1, 3, figsize=(8, 5))

    subsets = [len(not_match_ind_st1), len(not_match_ind_st2), len(ind_st1)]
    v = venn2(
        subsets=subsets,
        set_labels=["unit{}".format(cluster_id1), "unit{}".format(match2)],
        ax=axes[0],
    )
    v.get_patch_by_id("10").set_color("red")
    v.get_patch_by_id("01").set_color("blue")
    v.get_patch_by_id("11").set_color("goldenrod")
    axes[0].set_title(
        # f"{name1}{cluster_id1} + {name2}{match2}, "
        f"{match_frac.round(2)*100}% agreement"
    )

    which1 = np.flatnonzero(labels1 == cluster_id1)
    which2 = np.flatnonzero(labels2 == match2)

    # fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12,12))
    n_match_1 = len(ind_st1)
    n_unmatch_1 = len(not_match_ind_st1)
    match_unmatch_labels = np.r_[
        np.zeros(n_match_1, dtype=int),
        np.ones(n_unmatch_1, dtype=int),
    ]
    match_unmatch_spike_index = np.r_[
        spike_index1[which1[ind_st1]],
        spike_index1[which1[not_match_ind_st1]],
    ]
    match_unmatch_maxptps = np.r_[
        maxptps1[which1[ind_st1]],
        maxptps1[which1[not_match_ind_st1]],
    ]

    plot_waveforms_geom(
        0,
        [1],
        match_unmatch_labels,
        geom,
        channel_index,
        match_unmatch_spike_index,
        match_unmatch_maxptps,
        raw_bin=raw_bin,
        spikes_plot=spikes_plot,
        num_rows=3,
        ax=axes[1],
        colors=["goldenrod", "red"],
    )

    n_match_2 = len(ind_st2)
    n_unmatch_2 = len(not_match_ind_st2)
    match_unmatch_labels = np.r_[
        np.zeros(n_match_2, dtype=int),
        np.ones(n_unmatch_2, dtype=int),
    ]
    match_unmatch_spike_index = np.r_[
        spike_index2[which2[ind_st2]],
        spike_index2[which2[not_match_ind_st2]],
    ]
    match_unmatch_maxptps = np.r_[
        maxptps2[which2[ind_st2]],
        maxptps2[which2[not_match_ind_st2]],
    ]

    plot_waveforms_geom(
        0,
        [1],
        match_unmatch_labels,
        geom,
        channel_index,
        match_unmatch_spike_index,
        match_unmatch_maxptps,
        raw_bin=raw_bin,
        spikes_plot=spikes_plot,
        num_rows=3,
        ax=axes[2],
        colors=["goldenrod", "blue"],
    )

    return fig


def compute_spiketrain_agreement(st_1, st_2, delta_frames=12):
    # create figure for each match
    times_concat = np.concatenate((st_1, st_2))
    membership = np.concatenate(
        (np.ones(st_1.shape) * 1, np.ones(st_2.shape) * 2)
    )
    indices = times_concat.argsort()
    times_concat_sorted = times_concat[indices]
    membership_sorted = membership[indices]
    diffs = times_concat_sorted[1:] - times_concat_sorted[:-1]
    inds = np.where(
        (diffs <= delta_frames)
        & (membership_sorted[:-1] != membership_sorted[1:])
    )[0]
    if len(inds) > 0:
        inds2 = inds[np.where(inds[:-1] + 1 != inds[1:])[0]] + 1
        inds2 = np.concatenate((inds2, [inds[-1]]))
        times_matched = times_concat_sorted[inds2]
        # # find and label closest spikes
        ind_st1 = np.array(
            [np.abs(st_1 - tm).argmin() for tm in times_matched]
        )
        ind_st2 = np.array(
            [np.abs(st_2 - tm).argmin() for tm in times_matched]
        )
        not_match_ind_st1 = np.ones(st_1.shape[0], bool)
        not_match_ind_st1[ind_st1] = False
        not_match_ind_st1 = np.where(not_match_ind_st1)[0]
        not_match_ind_st2 = np.ones(st_2.shape[0], bool)
        not_match_ind_st2[ind_st2] = False
        not_match_ind_st2 = np.where(not_match_ind_st2)[0]
    else:
        ind_st1 = np.asarray([]).astype("int")
        ind_st2 = np.asarray([]).astype("int")
        not_match_ind_st1 = np.asarray([]).astype("int")
        not_match_ind_st2 = np.asarray([]).astype("int")

    return ind_st1, ind_st2, not_match_ind_st1, not_match_ind_st2
