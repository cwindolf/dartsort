from scipy.interpolate import interp1d, RectBivariateSpline


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

    def disp_at_s(self, t_s, depth_um=None):
        raise NotImplementedError

    def correct_s(self, t_s, depth_um):
        return depth_um - self.disp_at_s(t_s, depth_um)


class RigidMotionEstimate(MotionEstimate):
    def __init__(
        self,
        displacement,
        time_bin_edges_s=None,
        time_bin_centers_s=None,
        sampling_frequency=None,
    ):
        displacement = displacement.squeeze()

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
            fill_value="extrapolate",
        )

    def disp_at_s(self, t_s, depth_um=None):
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
            spatial_bin_edges_um=spatial_bin_centers_um,
        )

        self.lerp = RectBivariateSpline(
            self.spatial_bin_centers_um,
            self.time_bin_centers_s,
            self.displacement,
            kx=1,
            ky=1,
        )

    def disp_at_s(self, t_s, depth_um=None):
        return self.lerp(depth_um, t_s, grid=False)


class IdentityMotionEstimate(MotionEstimate):
    def __init__(self):
        super().__init__(None)

    def disp_at_s(self, t_s, depth_um=None):
        return 0.0


class ComposeMotionEstimates(MotionEstimate):
    def __init__(self, *motion_estimates):
        """Compose motion estimates, each estimated from the previous' corrections"""
        self.motion_estimates = motion_estimates
        super().__init__(None)

    def disp_at_s(self, t_s, depth_um=None):
        disp = 0
        if depth_um is None:
            depth_um = 0

        for me in self.motion_estimates:
            disp += me.disp_at_s(t_s, depth_um + disp)

        return disp
