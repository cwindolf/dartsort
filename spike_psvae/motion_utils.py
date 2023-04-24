import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample
from spikeinterface.sortingcomponents.motion_estimation import (
    get_windows as si_get_windows,
)


class MotionEstimate:
    def __init__(
        self,
        displacement,
        time_bin_edges_s=None,
        spatial_bin_edges_um=None,
        time_bin_centers_s=None,
        spatial_bin_centers_um=None,
    ):
        self.displacement = displacement
        self.time_bin_edges_s = time_bin_edges_s
        self.spatial_bin_edges_um = spatial_bin_edges_um

        self.time_bin_centers_s = time_bin_centers_s
        if time_bin_edges_s is not None:
            if time_bin_centers_s is None:
                self.time_bin_centers_s = 0.5 * (
                    time_bin_edges_s[1:] + time_bin_edges_s[:-1]
                )

        self.spatial_bin_centers_um = spatial_bin_centers_um
        if spatial_bin_edges_um is not None:
            if spatial_bin_centers_um is None:
                self.spatial_bin_centers_um = 0.5 * (
                    spatial_bin_edges_um[1:] + spatial_bin_edges_um[:-1]
                )

    def disp_at_s(self, t_s, depth_um=None, grid=False):
        raise NotImplementedError

    def correct_s(self, t_s, depth_um, grid=False):
        return depth_um - self.disp_at_s(t_s, depth_um, grid=grid)


class RigidMotionEstimate(MotionEstimate):
    def __init__(
        self,
        displacement,
        time_bin_edges_s=None,
        time_bin_centers_s=None,
    ):
        displacement = np.asarray(displacement).squeeze()

        assert displacement.ndim == 1
        if time_bin_edges_s is not None:
            assert (1 + displacement.shape[0],) == time_bin_edges_s.shape
        else:
            assert time_bin_centers_s is not None
            assert time_bin_centers_s.shape == displacement.shape

        super().__init__(
            displacement.squeeze(),
            time_bin_edges_s=time_bin_edges_s,
            time_bin_centers_s=time_bin_centers_s,
        )

        self.lerp = interp1d(
            self.time_bin_centers_s,
            self.displacement,
            bounds_error=False,
            fill_value=tuple(self.displacement[[0, -1]]),
        )

    def disp_at_s(self, t_s, depth_um=None, grid=False):
        assert not grid
        return self.lerp(t_s)


class NonrigidMotionEstimate(MotionEstimate):
    def __init__(
        self,
        displacement,
        time_bin_edges_s=None,
        time_bin_centers_s=None,
        spatial_bin_edges_um=None,
        spatial_bin_centers_um=None,
    ):
        assert displacement.ndim == 2
        if time_bin_edges_s is not None:
            time_bin_edges_s
            assert (1 + displacement.shape[1],) == time_bin_edges_s.shape
        else:
            assert time_bin_centers_s is not None
            assert (displacement.shape[1],) == time_bin_centers_s.shape
        if spatial_bin_edges_um is not None:
            assert (1 + displacement.shape[0],) == spatial_bin_edges_um.shape
        else:
            assert spatial_bin_centers_um is not None
            assert (displacement.shape[0],) == spatial_bin_centers_um.shape

        super().__init__(
            displacement,
            time_bin_edges_s=time_bin_edges_s,
            time_bin_centers_s=time_bin_centers_s,
            spatial_bin_edges_um=spatial_bin_edges_um,
            spatial_bin_centers_um=spatial_bin_centers_um,
        )

        # used below to disable RectBivariateSpline's extrapolation behavior
        # we'd rather fill in with the boundary value than make some line
        # going who knows where
        self.t_low = self.time_bin_centers_s.min()
        self.t_high = self.time_bin_centers_s.max()
        self.d_low = self.spatial_bin_centers_um.min()
        self.d_high = self.spatial_bin_centers_um.max()

        self.lerp = RectBivariateSpline(
            self.spatial_bin_centers_um,
            self.time_bin_centers_s,
            self.displacement,
            kx=1,
            ky=1,
        )

    def disp_at_s(self, t_s, depth_um=None, grid=False):
        return self.lerp(
            np.clip(depth_um, self.d_low, self.d_high),
            np.clip(t_s, self.t_low, self.t_high),
            grid=grid,
        )


class IdentityMotionEstimate(MotionEstimate):
    def __init__(self):
        super().__init__(None)

    def disp_at_s(self, t_s, depth_um=None):
        return 0.0


class ComposeMotionEstimates(MotionEstimate):
    def __init__(self, *motion_estimates):
        """Compose motion estimates, if each was estimated from the previous' corrections"""
        self.motion_estimates = motion_estimates
        super().__init__(None)

    def disp_at_s(self, t_s, depth_um=None):
        disp = 0
        if depth_um is None:
            depth_um = 0

        for me in self.motion_estimates:
            disp += me.disp_at_s(t_s, depth_um + disp)

        return disp


def get_motion_estimate(
    displacement,
    time_bin_edges_s=None,
    time_bin_centers_s=None,
    spatial_bin_edges_um=None,
    spatial_bin_centers_um=None,
    windows=None,
    window_weights=None,
    upsample_by_windows=False,
):
    displacement = np.asarray(displacement).squeeze()
    assert displacement.ndim <= 2
    assert any(a is not None for a in (time_bin_edges_s, time_bin_centers_s))

    # rigid case
    if displacement.ndim == 1:
        return RigidMotionEstimate(
            displacement,
            time_bin_edges_s=time_bin_edges_s,
            time_bin_centers_s=time_bin_centers_s,
        )
    assert any(
        a is not None for a in (spatial_bin_edges_um, spatial_bin_centers_um)
    )

    # linear interpolation nonrigid
    if not upsample_by_windows:
        return NonrigidMotionEstimate(
            displacement,
            time_bin_edges_s=time_bin_edges_s,
            time_bin_centers_s=time_bin_centers_s,
            spatial_bin_edges_um=spatial_bin_edges_um,
            spatial_bin_centers_um=spatial_bin_centers_um,
        )

    # upsample using the windows to spatial_bin_centers space
    if spatial_bin_centers_um is not None:
        D = spatial_bin_centers_um.shape[0]
    else:
        D = spatial_bin_edges_um.shape[0] - 1
    assert windows.shape == (displacement.shape[0], D)
    if window_weights is None:
        window_weights = np.ones_like(displacement)
    assert window_weights.shape == displacement.shape
    # precision weighted average
    normalizer = windows.T @ window_weights
    displacement_upsampled = (windows.T @ (P * window_weights)) / normalizer

    return NonrigidMotionEstimate(
        displacement_upsampled,
        time_bin_edges_s=time_bin_edges_s,
        time_bin_centers_s=time_bin_centers_s,
        spatial_bin_edges_um=spatial_bin_edges_um,
        spatial_bin_centers_um=spatial_bin_centers_um,
    )


def show_raster(
    raster, spatial_bin_edges_um, time_bin_edges_s, ax, **imshow_kwargs
):
    ax.imshow(
        raster,
        extent=(*time_bin_edges_s[[0, -1]], *spatial_bin_edges_um[[0, -1]]),
        origin="lower",
        **imshow_kwargs,
    )


def plot_me_traces(
    me, ax, offset=0, depths_um=None, label=False, **plot_kwargs
):
    if depths_um is None:
        depths_um = me.spatial_bin_centers_um

    for b, depth in enumerate(depths_um):
        disp = me.disp_at_s(me.time_bin_centers_s, depth_um=depth)
        lab = f"bin {b}" if label else None
        ax.plot(
            me.time_bin_centers_s,
            depth + offset + disp,
            label=lab,
            **plot_kwargs,
        )


def show_registered_raster(me, amps, depths, times, ax, **imshow_kwargs):
    depths_reg = me.correct_s(times, depths)
    raster, spatial_bin_edges_um, time_bin_edges_s = fast_raster(
        amps, depths_reg, times
    )
    ax.imshow(
        raster,
        extent=(*time_bin_edges_s[[0, -1]], *spatial_bin_edges_um[[0, -1]]),
        origin="lower",
        **imshow_kwargs,
    )


def show_dispmap(me, ax, spatial_bin_centers_um=None, **imshow_kwargs):
    if spatial_bin_centers_um is None:
        spatial_bin_centers_um = me.spatial_bin_centers_um

    dispmap = me.disp_at_s(
        me.time_bin_centers_s, spatial_bin_centers_um, grid=True
    )
    ax.imshow(
        dispmap,
        extent=(
            *me.time_bin_centers_s[[0, -1]],
            *spatial_bin_centers_um[[0, -1]],
        ),
        origin="lower",
        **imshow_kwargs,
    )


def get_bins(depths, times, bin_um, bin_s):
    spatial_bin_edges_um = np.arange(
        np.floor(depths.min()),
        np.ceil(depths.max()) + bin_um,
        bin_um,
    )
    time_bin_edges_s = np.arange(
        np.floor(times.min()),
        np.ceil(times.max()) + bin_s,
        bin_s,
    )
    return spatial_bin_edges_um, time_bin_edges_s


def get_windows(
    bin_um,
    spatial_bin_edges,
    geom,
    win_step_um,
    win_sigma_um,
    margin_um=0,
    win_shape="rect",
    zero_threshold=1e-5,
    rigid=False,
):
    if win_shape == "gaussian":
        win_sigma_um = win_sigma_um / 2
    windows, locs = si_get_windows(
        rigid=rigid,
        bin_um=bin_um,
        contact_pos=geom,
        spatial_bin_edges=spatial_bin_edges,
        margin_um=margin_um,
        win_step_um=win_step_um,
        win_sigma_um=win_sigma_um,
        win_shape=win_shape,
    )
    windows = np.array(windows)
    locs = np.array(locs)

    windows /= windows.sum(axis=1, keepdims=True)
    windows[windows < zero_threshold] = 0
    windows /= windows.sum(axis=1, keepdims=True)

    return windows, locs


def get_window_domains(windows):
    slices = []
    for w in windows:
        in_window = np.flatnonzero(w)
        slices.append(slice(in_window[0], in_window[-1] + 1))
    return slices


def fast_raster(
    amps,
    depths,
    times,
    bin_um=1,
    bin_s=1,
    spatial_bin_edges_um=None,
    time_bin_edges_s=None,
    amp_scale_fn=None,
    gaussian_smoothing_sigma_um=0,
    avg_in_bin=True,
    post_transform=None,
):
    if (spatial_bin_edges_um is None) or (time_bin_edges_s is None):
        _spatial_bin_edges_um, _time_bin_edges_s = get_bins(
            depths, times, bin_um, bin_s
        )
        if spatial_bin_edges_um is None:
            spatial_bin_edges_um = _spatial_bin_edges_um
        if time_bin_edges_s is None:
            time_bin_edges_s = _time_bin_edges_s

    if amp_scale_fn is None:
        weights = amps
    else:
        weights = amp_scale_fn(amps)

    if gaussian_smoothing_sigma_um:
        spatial_bin_edges_um_1um = np.arange(
            spatial_bin_edges_um[0],
            spatial_bin_edges_um[-1] + 1,
            1,
        )
        r_up = np.histogram2d(
            depths,
            times,
            bins=(spatial_bin_edges_um_1um, time_bin_edges_s),
            weights=weights,
        )[0]
        if avg_in_bin:
            r_up /= np.maximum(
                1,
                np.histogram2d(
                    depths,
                    times,
                    bins=(spatial_bin_edges_um_1um, time_bin_edges_s),
                )[0],
            )

        r_up = gaussian_filter1d(r_up, gaussian_smoothing_sigma_um / bin_um)
        r = resample(r_up, spatial_bin_edges_um.size - 1)
    else:
        r = np.histogram2d(
            depths,
            times,
            bins=(spatial_bin_edges_um, time_bin_edges_s),
            weights=weights,
        )[0]
        if avg_in_bin:
            r /= np.maximum(
                1,
                np.histogram2d(
                    depths,
                    times,
                    bins=(spatial_bin_edges_um, time_bin_edges_s),
                )[0],
            )

    if post_transform is not None:
        r = post_transform(r)

    return r, spatial_bin_edges_um, time_bin_edges_s
