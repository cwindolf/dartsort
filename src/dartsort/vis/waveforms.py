import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
import numpy as np
import torch


def geomplot(
    waveforms,
    max_channels=None,
    channel_index=None,
    channels=None,
    geom=None,
    ax=None,
    z_extension=1.0,
    x_extension=0.8,
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
    return_chans=False,
    linestyles=None,
    linewidths=None,
    randomize=False,
    random_seed=0,
    **plot_kwargs,
):
    """Plot waveforms according to geometry using channel index"""
    assert geom is not None
    ax = ax or plt.gca()

    # args to numpy if necessary
    if torch.is_tensor(waveforms):
        waveforms = waveforms.numpy(force=True)
    if torch.is_tensor(max_channels):
        max_channels = max_channels.numpy(force=True)
    if torch.is_tensor(channels):
        channels = channels.numpy(force=True)
    if torch.is_tensor(geom):
        geom = geom.numpy(force=True)
    if torch.is_tensor(c):
        c = c.numpy(force=True)

    # randomize if asked to
    if randomize:
        rg = np.random.default_rng(random_seed)
        o = rg.permutation(len(waveforms))
        if waveforms is not None:
            waveforms = waveforms[o]
        if max_channels is not None:
            max_channels = max_channels[o]
        if channels is not None:
            channels = channels[o]
        if c is not None and hasattr(c, "__len__") and len(c) == len(o):
            c = c[o]
        if colors is not None and len(colors) == len(o):
            colors = colors[o]

    # -- validate shapes
    if waveforms.ndim == 2:
        waveforms = waveforms[None]
    assert waveforms.ndim == 3
    if c is not None and (hasattr(c, "__len__") and len(c) > 1):
        assert len(c) == len(waveforms)
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
            raise ValueError(
                f"Bad shapes: {waveforms.shape=}, {max_channels.shape=}, {C=}"
            )
        channels = channel_index[max_channels]
    else:
        n_channels = geom[np.isfinite(geom).all(1)].shape[0]
        T = waveforms.shape[1]
        assert channels.shape[0] == waveforms.shape[0]
        assert channels.shape[1] == waveforms.shape[-1]

    # -- figure out units for plotting
    valid = np.isfinite(geom).all(1)
    z_uniq, z_ix = np.unique(geom[valid, 1], return_inverse=True)
    for i in z_ix:
        x_uniq = np.unique(geom[valid][z_ix == i, 0])
        if x_uniq.size > 1:
            break
    else:
        x_uniq = np.unique(geom[valid, 0])
    inter_chan_x = 1
    if x_uniq.size > 1:
        inter_chan_x = x_uniq[1] - x_uniq[0]
    inter_chan_z = z_uniq[1] - z_uniq[0]
    max_abs_amp = max_abs_amp or np.nanmax(np.abs(waveforms))
    geom_scales = [
        T / inter_chan_x / x_extension,
        max_abs_amp / inter_chan_z / z_extension,
    ]
    geom_plot = geom[valid] * geom_scales
    t_domain = np.linspace(-T // 2, T // 2, num=T)

    # -- and, plot
    draw = []
    draw_colors = []
    unique_chans = set()
    xmin, xmax = np.inf, -np.inf
    if linestyles is not None:
        ls = []
        plot_kwargs["linestyles"] = ls
    if linewidths is not None:
        lw = []
        plot_kwargs["linewidths"] = lw
    for j, (wf, chans) in enumerate(zip(waveforms, channels)):
        for i, cc in enumerate(chans):
            if cc == n_channels:
                continue

            draw.append(np.c_[geom_plot[cc, 0] + t_domain, geom_plot[cc, 1] + wf[:, i]])
            xmin = min(geom_plot[cc, 0] + t_domain.min(), xmin)
            xmax = max(geom_plot[cc, 0] + t_domain.max(), xmax)
            unique_chans.add(cc)
            if colors is not None:
                draw_colors.append(to_rgba(colors[j]))
            elif color is not None:
                draw_colors.append(to_rgba(color))
            elif c is not None:
                cj = c[j] if (hasattr(c, "__len__") and len(c) > 1) else c
                draw_colors.append(to_rgba(cj))
            if linestyles is not None:
                ls.append(linestyles[j])
            if linewidths is not None:
                lw.append(linewidths[j])
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
    lines = LineCollection(
        np.array(draw),
        colors=np.array(draw_colors) if draw_colors else None,
        **plot_kwargs,
    )
    lines = ax.add_collection(lines)
    if annotate_z:
        unique_z = np.unique(geom[:, 1][list(unique_chans)])
        unique_zp = np.unique(geom_plot[:, 1][list(unique_chans)])
        for z, zp in zip(unique_z, unique_zp):
            ax.text(
                xmin,
                zp,
                f"{z:f}".rstrip("0").rstrip("."),
                size=6,
                color="gray",
                clip_on=True,
            )

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
    elif zlim is False:
        pass

    if return_chans:
        return lines, unique_chans

    return lines
