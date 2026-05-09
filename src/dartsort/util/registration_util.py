import numpy as np
import torch
from dredge.dredge_ap import register as dredge_register
from dredge.motion_util import MotionEstimate, speed_limit_filter
from spikeinterface.core import BaseRecording, Motion

from .data_util import DARTsortSorting
from .internal_config import MotionEstimationConfig, default_motion_estimation_cfg


def dredge_estimate_motion(
    recording: BaseRecording,
    sorting: DARTsortSorting,
    motion_cfg: MotionEstimationConfig = default_motion_estimation_cfg,
    localizations_dataset_name="point_source_localizations",
    amplitudes_dataset_name="denoised_ptp_amplitudes",
    device: torch.device | None = None,
) -> MotionEstimate | None:
    if not motion_cfg.do_motion_estimation:
        return None

    x = getattr(sorting, localizations_dataset_name)[:, 0]
    z = getattr(sorting, localizations_dataset_name)[:, 2]
    a = getattr(sorting, amplitudes_dataset_name)
    geom = recording.get_channel_locations()
    xmin = geom[:, 0].min() - motion_cfg.probe_boundary_padding_um
    xmax = geom[:, 0].max() + motion_cfg.probe_boundary_padding_um
    zmin = geom[:, 1].min() - motion_cfg.probe_boundary_padding_um
    zmax = geom[:, 1].max() + motion_cfg.probe_boundary_padding_um
    valid = x == x.clip(xmin, xmax)
    valid &= z == z.clip(zmin, zmax)
    if motion_cfg.min_amplitude:
        valid &= a >= motion_cfg.min_amplitude
    valid = np.flatnonzero(valid)

    # features for registration
    z = z[valid]
    t_s = getattr(sorting, "times_seconds")
    t_s = t_s[valid]
    a = a[valid]

    # run registration
    print(motion_cfg.smoothing_um)
    dredge_motion_est, _ = dredge_register(
        amps=a,
        depths_um=z,
        times_s=t_s,
        rigid=motion_cfg.rigid,
        bin_um=motion_cfg.spatial_bin_length_um,
        bin_s=motion_cfg.temporal_bin_length_s,
        win_step_um=motion_cfg.window_step_um,
        weights_threshold_low=motion_cfg.weight_threshold,
        weights_threshold_high=motion_cfg.weight_threshold,
        win_scale_um=motion_cfg.window_scale_um,
        win_margin_um=motion_cfg.window_margin_um,
        max_disp_um=motion_cfg.max_disp_um,
        max_dt_s=motion_cfg.max_dt_s,
        mincorr=motion_cfg.correlation_threshold,
        gaussian_smoothing_sigma_um=motion_cfg.smoothing_um,
        gaussian_smoothing_sigma_s=motion_cfg.smoothing_s,
        device=device,
    )
    dredge_motion_est = speed_limit_filter(
        dredge_motion_est,
        band_width=motion_cfg.median_neighborhood_bins,
        band_limit=motion_cfg.max_dist_from_median_um,
        speed_limit_um_per_s=motion_cfg.speed_limit_um_per_s,
    )

    return dredge_motion_est


def dredge_to_si(dredge_motion_est: MotionEstimate) -> Motion:
    disp = dredge_motion_est.displacement
    t = dredge_motion_est.time_bin_centers_s
    if disp.ndim == 1:
        # rigid case
        return Motion(
            displacement=disp[:, None],
            temporal_bins_s=t,
            spatial_bins_um=np.zeros(1),
        )
    else:
        # TODO
        raise NotImplementedError
