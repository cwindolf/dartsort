import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba


def geomplot(
    waveforms,
    colors=None,
    max_channels=None,
    channel_index=None,
    channels=None,
    geom=None,
    ax=None,
    z_extension=1.0,
    x_extension=0.95,
    show_zero=True,
    show_zero_kwargs=None,
    max_abs_amp=None,
    show_chan_label=False,
    annotate_z=True,
    chan_labels=None,
    xlim_factor=1,
    subar=False,
    msbar=False,
    bar_color="k",
    bar_background="w",
    zlim="tight",
    c=None,
    color=None,
    colors=None,
    **plot_kwargs,
):
    """Plot waveforms according to geometry using channel index"""
    assert geom is not None
    ax = ax or plt.gca()

    # -- validate shapes
    if waveforms.ndim == 2:
        waveforms = waveforms[None]
    assert waveforms.ndim == 3
    if channels is None:
        if max_channels is None and channel_index is None:
            max_channels = np.zeros(waveforms.shape[0], dtype=int)
            channel_index = (
                np.arange(geom.shape[0])[None, :]
                * np.ones(geom.shape[0], dtype=int)[:, None]
            )
        max_channels = np.atleast_1d(max_channels)
        n_channels, C = channel_index.shape
        assert geom.shape == (n_channels, 2)
        T = waveforms.shape[1]
        if waveforms.shape != (*max_channels.shape, T, C):
            raise ValueError(f"Bad shapes: {waveforms.shape=}, {max_channels.shape=}, {C=}")
        channels = channel_index[max_channels]
    else:
        n_channels = geom.shape[0]
        T = waveforms.shape[1]
        assert channels.shape[0] == waveforms.shape[0]
        assert channels.shape[1] == waveforms.shape[-1]

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
    if rearrange:
        geom_plot = np.array(np.meshgrid(np.arange(n_chans_per_line), np.arange(geom.shape[0]/n_chans_per_line))).reshape(2, -1).T
        geom_plot = geom_plot*np.array([inter_chan_x, inter_chan_z])[None]
        geom_plot = geom_plot * geom_scales
    t_domain = np.linspace(-T // 2, T // 2, num=T)

    # -- and, plot
    draw = []
    draw_colors = []
    unique_chans = set()
    xmin, xmax = np.inf, -np.inf
    for j, (wf, chans) in enumerate(zip(waveforms, channels)):
        for i, c in enumerate(chans):
            if c == n_channels:
                continue

            draw.append(np.c_[geom_plot[c, 0] + t_domain, geom_plot[c, 1] + wf[:, i]])
            xmin = min(geom_plot[c, 0] + t_domain.min(), xmin)
            xmax = max(geom_plot[c, 0] + t_domain.max(), xmax)
            unique_chans.add(c)
            if colors is not None:
                draw_colors.append(to_rgba(colors[j]))
            elif color is not None:
                draw_colors.append(to_rgba(color))
            elif c is not None:
                draw_colors.append(to_rgba(c))
    dx = xmax - xmin
    xpad = dx / 2 - xlim_factor * dx / 2
    ax.set_xlim([xmin + xpad, xmax - xpad])

    ann_offset = np.array([0, 0.33 * inter_chan_z]) * geom_scales
    chan_labels = (
        chan_labels if chan_labels is not None else list(map(str, range(len(geom))))
    )
    for c in unique_chans:
        if show_zero:
            if show_zero_kwargs is None:
                show_zero_kwargs = dict(color="gray", lw=0.8, linestyle="--")
            ax.axhline(geom_plot[c, 1], **show_zero_kwargs)
        if show_chan_label:
            ax.annotate(chan_labels[c], geom_plot[c] + ann_offset, size=6, color="gray")
    lines = ax.plot(*draw, **plot_kwargs)
    if colors is not None:
        for line, color_line in zip(lines,colors):
            line.set_color(color_line)

    if subar:
        if isinstance(subar, bool):
            # auto su if True, but you can pass int/float too.
            subars = (1, 2, 5, 10, 20, 30, 50)
            for j in range(len(subars)):
                if subars[j] > 1.2 * max_abs_amp:
                    break
            subar = subars[max(0, j - 1)]
        min_z = min(geom_plot[c, 1] for c in unique_chans)
        if msbar:
            min_z += max_abs_amp
        if bar_background:
            ax.add_patch(
                Rectangle(
                    [
                        xmax - T // 4 - 2,
                        min_z - max_abs_amp / 2 - subar / 10,
                    ],
                    4 + 7 + 2 + T // 8,
                    subar + subar / 5,
                    fc=bar_background,
                    zorder=11,
                    alpha=0.8,
                )
            )
        ax.add_patch(
            Rectangle(
                [
                    xmax - T // 4,
                    min_z - max_abs_amp / 2,
                ],
                4,
                subar,
                fc=bar_color,
                zorder=12,
            )
        )
        ax.text(
            xmax - T // 4 + 5,
            min_z - max_abs_amp / 2 + subar / 2,
            f"{subar} s.u.",
            transform=ax.transData,
            fontsize=5,
            ha="left",
            va="center",
            rotation=-90,
            color=bar_color,
            zorder=12,
        )

    if msbar:
        min_z = min(geom_plot[c, 1] for c in unique_chans)
        ax.plot(
            [
                geom_plot[:, 0].max() - 30,
                geom_plot[:, 0].max(),
            ],
            2 * [min_z - max_abs_amp],
            color=bar_color,
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
            color=bar_color,
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
            ax.set_ylim([min_z - max_abs_amp * zlim, max_z + max_abs_amp * zlim])

    return lines


def geomplot_max_chan_overtime(
    waveforms_list,
    wfs_stacked,
    colors=None,
    geom=None,
    ax=None,
    z_extension=1.0,
    x_extension=0.95,
    show_zero=True,
    show_zero_kwargs=None,
    max_abs_amp=None,
    show_chan_label=False,
    chan_labels=None,
    xlim_factor=1,
    subar=False,
    msbar=False,
    bar_color="k",
    bar_background="w",
    zlim="tight",
    rearrange=True,
    n_chans_per_line=4,
    **plot_kwargs,
):
    """Plot waveforms according to geometry using channel index"""
    assert geom is not None
    ax = ax or plt.gca()

    # -- validate shapes
    if wfs_stacked.ndim == 2:
        wfs_stacked = wfs_stacked[None]
    assert wfs_stacked.ndim == 3
    T = wfs_stacked.shape[1]

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
    max_abs_amp = max_abs_amp or np.nanmax(np.abs(wfs_stacked))
    geom_scales = [
        T / inter_chan_x / x_extension,
        max_abs_amp / inter_chan_z / z_extension,
    ]
    geom_plot = geom * geom_scales
    if rearrange:
        geom_plot = np.array(np.meshgrid(np.arange(n_chans_per_line), np.arange(geom.shape[0]/n_chans_per_line))).reshape(2, -1).T
        geom_plot = geom_plot*np.array([inter_chan_x, inter_chan_z])[None]
        geom_plot = geom_plot * geom_scales
    t_domain = np.linspace(-T // 2, T // 2, num=T)

    # -- and, plot
    draw = []
    unique_chans = set()
    xmin, xmax = np.inf, -np.inf
    for k in range(len(waveforms_list)):
        for wf in waveforms_list[k]: 
            draw.append(geom_plot[k, 0] + t_domain)
            draw.append(geom_plot[k, 1] + wf[:, 0])
        xmin = min(geom_plot[k, 0] + t_domain.min(), xmin)
        xmax = max(geom_plot[k, 0] + t_domain.max(), xmax)
        unique_chans.add(k)

    dx = xmax - xmin
    ax.set_xlim(
        [
            xmin + dx / 2 - xlim_factor * dx / 2,
            xmax - dx / 2 + xlim_factor * dx / 2,
        ]
    )

    ann_offset = np.array([0, 0.33 * inter_chan_z]) * geom_scales
    chan_labels = (
        chan_labels if chan_labels is not None else list(map(str, range(len(geom))))
    )

    for c in unique_chans:
        if show_zero:
            if show_zero_kwargs is None:
                show_zero_kwargs = dict(color="gray", lw=0.8, linestyle="--")
            ax.axhline(geom_plot[c, 1], **show_zero_kwargs)
        if show_chan_label:
            ax.annotate(chan_labels[c], geom_plot[c] + ann_offset, size=6, color="gray")
    lines = ax.plot(*draw, **plot_kwargs)
    if colors is not None:
        for line, color_line in zip(lines,colors):
            line.set_color(color_line)

    if subar:
        if isinstance(subar, bool):
            # auto su if True, but you can pass int/float too.
            subars = (1, 2, 5, 10, 20, 30, 50)
            for j in range(len(subars)):
                if subars[j] > 1.2 * max_abs_amp:
                    break
            subar = subars[max(0, j - 1)]
        min_z = min(geom_plot[c, 1] for c in unique_chans)
        if msbar:
            min_z += max_abs_amp
        if bar_background:
            ax.add_patch(
                Rectangle(
                    [
                        xmax - T // 4 - 2,
                        min_z - max_abs_amp / 2 - subar / 10,
                    ],
                    4 + 7 + 2 + T // 8,
                    subar + subar / 5,
                    fc=bar_background,
                    zorder=11,
                    alpha=0.8,
                )
            )
        ax.add_patch(
            Rectangle(
                [
                    xmax - T // 4,
                    min_z - max_abs_amp / 2,
                ],
                4,
                subar,
                fc=bar_color,
                zorder=12,
            )
        )
        ax.text(
            xmax - T // 4 + 5,
            min_z - max_abs_amp / 2 + subar / 2,
            f"{subar} s.u.",
            transform=ax.transData,
            fontsize=5,
            ha="left",
            va="center",
            rotation=-90,
            color=bar_color,
            zorder=12,
        )

    if msbar:
        min_z = min(geom_plot[c, 1] for c in unique_chans)
        ax.plot(
            [
                geom_plot[:, 0].max() - 30,
                geom_plot[:, 0].max(),
            ],
            2 * [min_z - max_abs_amp],
            color=bar_color,
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
            color=bar_color,
        )

    if zlim is None:
        ax.autoscale_view()
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
            ax.set_ylim([min_z - max_abs_amp * zlim, max_z + max_abs_amp * zlim])

    return lines
