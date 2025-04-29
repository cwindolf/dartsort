import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import ResizeEvent
import numpy as np
from tqdm.auto import trange
import seaborn as sns

from .colors import glasbey1024
from ..localize.localize_util import localize_waveforms

glasbey1024a = np.c_[glasbey1024, np.ones(len(glasbey1024))]
_t = np.array([1.0, 1, 1, 0])
_LW = 1
_MAX_SC = 512


class RecordingAnimation:
    def __init__(
        self,
        recording,
        sorting,
        template_data,
        offset_frames=100,
        show_frames=500,
        zlim_start=None,
        zlim_target=None,
        zlim_hold_s=0.01,
        zlim_target_s=0.01,
        contact_wh=(15.0, 15.0),
        soma_size=10.0,
        soma_far_dist=15.0,
        vmin=-5,
        vmax=5,
        cmap="seismic",
        recording_frames_per_second=1000,
        start_frame=0,
        end_frame=None,
    ):
        self.geom = recording.get_channel_locations()
        self.recording = recording
        self.sorting = sorting
        self.template_data = template_data
        self.trough_offset_samples = self.template_data.trough_offset_samples
        self.spike_length_samples = self.template_data.spike_length_samples

        self.contact_wh = contact_wh
        self.vmin = vmin
        self.vmax = vmax
        self.cmap = plt.get_cmap(cmap)

        self.soma_size = soma_size
        self.soma_far_dist = soma_far_dist

        self.probe_zmin = self.geom[:, 1].min()
        self.probe_zmax = self.geom[:, 1].max()
        self.zlim_start = zlim_start
        self.zlim_target = zlim_target
        self.zlim_target_s = zlim_target_s
        self.zlim_hold_s = zlim_hold_s
        self.zmargin = 3 * contact_wh[1]
        self.show_frames = show_frames
        self.offset_frames = offset_frames

        self.recording_frames_per_second = recording_frames_per_second
        self.start_frame = start_frame
        if end_frame is None:
            end_frame = recording.get_num_frames() - self.spike_length_samples
        self.end_frame = end_frame

        self.unit_mainchan_signal = unit_signal_traces(
            recording, sorting, template_data
        )
        self.unit_maxchans = np.abs(template_data.templates).max(1).argmax(1)
        self.signal = convolve(recording, sorting, template_data)
        locs = localize_waveforms(template_data.templates, self.geom)
        self.unit_xyz = np.c_[locs["x"], locs["y"], locs["z_abs"]]

        # initialized later
        self._has_drawn = False
        self.figure = None
        self.soma_patches = None
        self.contact_patches = None
        self.recording_ax = None
        self.signal_ax = None
        self.probe_ax = None
        self.recording_im = None
        self.signal_im = None
        self.signal_sc = None

    def clear(self):
        self.set_drawables()
        self.vlines = None
        self.signal_im = self.recording_im = self.signal_sc = None
        self.artists = None
        self._has_drawn = False

    def set_drawables(
        self,
        figure=None,
        soma_patches=None,
        contact_patches=None,
        recording_ax=None,
        signal_ax=None,
        probe_ax=None,
    ):
        self.figure = figure
        self.soma_patches = soma_patches
        self.contact_patches = contact_patches
        self.recording_ax = recording_ax
        self.signal_ax = signal_ax
        self.probe_ax = probe_ax

    def initialize_figure(self, figsize=(5, 5)):
        figure = plt.Figure(figsize=figsize, constrained_layout=True)
        axes = figure.subplot_mosaic("ab\nac", gridspec_kw=dict(width_ratios=[1, 3]))
        left_ax = axes["a"]
        right_axes = [axes["b"], axes["c"]]
        # left_column, right_column = figure.subfigures(ncols=2, width_ratios=[2, 6])
        # left_ax = left_column.subplots()

        left_ax.set_aspect(1.0)
        shank, contact_patches = draw_probe(left_ax, self.geom, *self.contact_wh)
        soma_patches = [
            draw_neuron(
                left_ax,
                *self.unit_xyz[j],
                size=self.soma_size,
                far_dist=self.soma_far_dist,
                edgecolor=glasbey1024[j],
            )
            for j in range(len(self.template_data.templates))
        ]
        soma_patches = PatchCollection(soma_patches, match_original=True)
        left_ax.add_collection(soma_patches)
        sns.despine(ax=left_ax, trim=True)
        left_ax.set_xlim(
            [
                self.geom[:, 0].min() - 2 * self.contact_wh[0],
                self.geom[:, 0].max() + 3 * self.contact_wh[0],
            ]
        )
        left_ax.set_xticks([self.geom[:, 0].min(), self.geom[:, 0].max()])
        left_ax.set_xlabel("x (um)")
        left_ax.set_ylabel("z (um)")

        vlines = []
        for ax in right_axes:
            ax.set_ylabel("z (um)")
            ax.set_xlabel("time (s)")
            vlines.append(ax.axvline(self.offset_frames, color="k", lw=0.8, zorder=11))
        self.vlines = vlines

        self.set_drawables(
            figure=figure,
            soma_patches=soma_patches,
            contact_patches=contact_patches,
            recording_ax=right_axes[0],
            signal_ax=right_axes[1],
            probe_ax=left_ax,
        )
        self.artists = [shank, contact_patches, soma_patches, *vlines]

    def draw_frame(self, frame_ix):
        self.probe_ax.set_ylim(self.zlim(frame_ix))
        sns.despine(ax=self.probe_ax, trim=True)
        self.update_contacts(frame_ix)
        self.update_somas(frame_ix)
        imr = self.update_recording(frame_ix)
        ims = self.update_signal(frame_ix)
        for ax in (self.recording_ax, self.signal_ax):
            ax.set_xlim(
                [
                    frame_ix / self.recording.sampling_frequency,
                    (frame_ix + self.show_frames) / self.recording.sampling_frequency,
                ]
            )
            dt = 0.004
            ft0 = frame_ix / self.recording.sampling_frequency
            ft1 = (frame_ix + self.show_frames) / self.recording.sampling_frequency
            t0 = dt * np.ceil(ft0 / dt)
            t1 = dt * np.floor((ft1 - dt) / dt)
            t1 = max(t0, t1)
            ax.set_xticks(np.arange(t0, t1 + 1e-6, dt))

        z0, z1 = self.zlim(frame_ix)
        z0 = self.geom[:, 1][self.geom[:, 1] > z0].min()
        z1 = self.geom[:, 1][self.geom[:, 1] < z1].max()
        self.signal_ax.set_ylim([z0, z1])

        self._has_drawn = True
        for vl in self.vlines:
            vl.set_xdata(
                2
                * [(frame_ix + self.offset_frames) / self.recording.sampling_frequency]
            )
        return self.artists + [imr, ims]

    def animate(self, start_frame=None, end_frame=None):
        if self.figure is None:
            self.initialize_figure()
        # if not self._has_drawn:
        #     self.draw_frame(0)

        if start_frame is None:
            start_frame = self.start_frame
        if end_frame is None:
            end_frame = self.end_frame

        # artists = []
        # for frame in :
        #     frame_artists = self.draw_frame(frame)
        #     artists.append(frame_artists)
        with trange(start_frame, end_frame, desc="Frames") as pbar:
            anim = FuncAnimation(
                self.figure,
                self.draw_frame,
                frames=pbar,
                cache_frame_data=False,
                blit=False,
                interval=0,
            )

        # anim = ArtistAnimation(self.figure, artists, interval=50, blit=False)

        return anim

    def update_somas(self, frame_ix):
        signals = self.unit_mainchan_signal[frame_ix + self.offset_frames]
        facecolors = self.to_color(signals)
        self.soma_patches.set_facecolor(facecolors)

    def update_contacts(self, frame_ix):
        signals = self.recording.get_traces(
            start_frame=frame_ix + self.offset_frames,
            end_frame=frame_ix + self.offset_frames + 1,
        )
        facecolors = self.to_color(signals.ravel())
        self.contact_patches.set_facecolor(facecolors)

    def update_recording(self, frame_ix):
        ac = self.active_channels(frame_ix)
        chunk = self.recording.get_traces(
            start_frame=frame_ix, end_frame=frame_ix + self.show_frames, channel_ids=ac
        )
        if self.recording_im is not None:
            self.recording_im.set_data(chunk[:, ::-1].T)
            self.recording_im.set_extent(self.extent(frame_ix))
        self.recording_im = self.recording_ax.imshow(
            chunk[:, ::-1].T,
            extent=self.extent(frame_ix),
            vmin=self.vmin,
            vmax=self.vmax,
            cmap=self.cmap,
            interpolation="none",
            aspect="auto",
            # animated=self._has_drawn,
        )
        return self.recording_im

    def update_signal(self, frame_ix):
        ac = self.active_channels(frame_ix)
        chunk = self.signal[frame_ix : frame_ix + self.show_frames, ac]

        if self.signal_im is not None:
            self.signal_im.set_data(chunk.T)
            self.signal_im.set_extent(self.extent(frame_ix))
            return self.signal_im
        self.signal_im = self.signal_ax.imshow(
            chunk.T,
            extent=self.extent(frame_ix),
            vmin=self.vmin,
            vmax=self.vmax,
            cmap=self.cmap,
            interpolation="none",
            aspect="auto",
            origin="lower",
            # animated=self._has_drawn,
        )

        self.signal_sc = self.signal_ax.scatter(
            self.sorting.times_samples / self.recording.sampling_frequency,
            self.geom[:, 1][self.unit_maxchans[self.sorting.labels]],
            edgecolor=glasbey1024a[self.sorting.labels],
            marker="o",
            s=10,
            zorder=10,
            facecolor=_t,
            linewidth=_LW,
        )
        return self.signal_im

    def active_channels(self, frame_ix):
        zlim = self.zlim(frame_ix)
        z = self.geom[:, 1]
        return np.flatnonzero(z.clip(*zlim) == z)

    def zlim(self, frame_ix):
        zmin = self.probe_zmin - self.zmargin
        zmax = self.probe_zmax + self.zmargin
        if self.zlim_start is None and self.zlim_target is None:
            return [zmin, zmax]

        frame_s = frame_ix / self.recording.sampling_frequency
        if frame_s < self.zlim_hold_s:
            return [zmin, zmax]
        frame_s = frame_s - self.zlim_hold_s

        if self.zlim_start is None:
            zlim_start = zmin, zmax

        zmin0, zmax0 = zlim_start
        zmin1, zmax1 = self.zlim_target
        if frame_s > self.zlim_target_s:
            return [zmin1, zmax1]
        lerp1 = frame_s / self.zlim_target_s
        lerp0 = 1 - lerp1
        return [lerp0 * zmin0 + lerp1 * zmin1, lerp0 * zmax0 + lerp1 * zmax1]

    def extent(self, frame_ix):
        t0 = frame_ix / self.recording.sampling_frequency
        t1 = (frame_ix + self.show_frames - 1) / self.recording.sampling_frequency
        return [t0, t1, *self.zlim(frame_ix)]

    def to_color(self, voltages):
        v = voltages - self.vmin
        v /= self.vmax - self.vmin
        return self.cmap(v)


def draw_neuron(ax, x, y, z, size=10.0, far_dist=100.0, edgecolor="k"):
    radius = size * np.sqrt(far_dist / y)
    soma = RegularPolygon(
        (x, z),
        numVertices=3,
        radius=radius,
        facecolor=_t,
        edgecolor=edgecolor,
        linewidth=_LW,
        zorder=1 + np.around(2 + 100 * y),
    )
    for theta in np.arange(np.pi / 2, 2 * np.pi, 2 * np.pi / 3):
        vx = x + radius * np.cos(theta)
        vz = z + radius * np.sin(theta)
        vz1 = vz + radius * (2 * (vz > z) - 1)
        ax.plot(
            [vx, vx], [vz, vz1], lw=_LW, zorder=np.around(2 + 100 * y), color=edgecolor
        )
    return soma


def draw_probe(ax, geom, contact_w=10.0, contact_h=10.0, shank_color="lightgray"):
    xmin, zmin = geom.min(axis=0)
    xmax, zmax = geom.max(axis=0)
    (shank,) = ax.fill(
        [
            xmin - contact_w,
            xmin - contact_w,
            (xmin + xmax) / 2,
            xmax + 2 * contact_w,
            xmax + 2 * contact_w,
        ],
        [zmax + 2 * contact_h, zmin, zmin - 4 * contact_h, zmin, zmax + 2 * contact_h],
        facecolor=shank_color,
        edgecolor="k",
        linewidth=_LW,
        zorder=0,
    )

    # return contact patches
    contact_patches = [
        Rectangle(
            xy,
            contact_w,
            contact_h,
            edgecolor="k",
            linewidth=_LW,
            facecolor=_t,
            zorder=1,
            joinstyle="bevel",
        )
        for xy in geom
    ]
    contact_patches = PatchCollection(contact_patches, match_original=True)
    ax.add_collection(contact_patches)
    return shank, contact_patches


def unit_signal_traces(recording, sorting, template_data):
    trough_offset_samples = template_data.trough_offset_samples
    temps = template_data.templates

    mcs = np.argmax(np.abs(temps).max(axis=1), axis=1)
    tmcs = temps[np.arange(len(mcs)), :, mcs]
    tmcs = tmcs.astype(np.float16)

    time_ix = np.arange(temps.shape[1]) - trough_offset_samples
    time_ix = sorting.times_samples[:, None] + time_ix

    traces = np.zeros((recording.get_num_frames(), len(temps)), dtype=np.float16)
    np.add.at(
        traces,
        (time_ix, sorting.labels[:, None]),
        tmcs[sorting.labels],
    )
    return traces


def convolve(recording, sorting, template_data, chunklen=2**12):
    traces = np.zeros(
        (recording.get_num_frames(), recording.get_num_channels()), dtype=np.float16
    )
    temps = template_data.templates.astype(np.float16)

    t_rel_ix = np.arange(template_data.spike_length_samples)
    t_rel_ix -= template_data.trough_offset_samples
    chan_ix = np.arange(traces.shape[1])

    t = sorting.times_samples
    for chunk_start in trange(0, traces.shape[0], chunklen, desc="Convolve"):
        in_chunk = t == t.clip(chunk_start, chunk_start + chunklen - 1)
        bt = t[in_chunk]
        bl = sorting.labels[in_chunk]
        btemps = temps[bl]
        tix = bt[:, None, None] + t_rel_ix[None, :, None]
        np.add.at(traces, (tix, chan_ix[None, None]), btemps)

    return traces
