from typing import Optional

import numpy as np

try:
    from dredge import dredge_ap

    have_dredge = True
except ImportError:
    have_dredge = False
    pass


def estimate_motion(
    recording,
    sorting,
    do_motion_estimation=True,
    probe_boundary_padding_um=100.0,
    spatial_bin_length_um: float = 1.0,
    temporal_bin_length_s: float = 1.0,
    window_step_um: float = 400.0,
    window_scale_um: float = 450.0,
    window_margin_um: Optional[float] = None,
    max_dt_s: float = 0.1,
    max_disp_um: Optional[float] = None,
    localizations_dataset_name="point_source_localizations",
    amplitudes_dataset_name="denoised_ptp_amplitudes",
):
    if not do_motion_estimation:
        return None

    if not have_dredge:
        raise ValueError("Please install DREDge to use motion estimation.")

    x = getattr(sorting, localizations_dataset_name)[:, 0]
    z = getattr(sorting, localizations_dataset_name)[:, 1]
    geom = recording.get_channel_locations()
    xmin = geom[:, 0].min() - probe_boundary_padding_um
    xmax = geom[:, 0].max() + probe_boundary_padding_um
    zmin = geom[:, 1].min() - probe_boundary_padding_um
    zmax = geom[:, 1].max() + probe_boundary_padding_um
    xvalid = x == np.clip(x, xmin, xmax)
    zvalid = z == np.clip(z, zmin, zmax)
    valid = np.flatnonzero(xvalid & zvalid)

    # features for registration
    z = z[valid]
    t_s = sorting.times_seconds[valid]
    a = getattr(sorting, amplitudes_dataset_name)[valid]

    # run registration
    motion_est, info = dredge_ap.register(
        a,
        z,
        t_s,
        window_step_um=window_step_um,
        bin_um=spatial_bin_length_um,
        bin_s=temporal_bin_length_s,
        window_scale_um=window_scale_um,
        window_margin_um=window_margin_um,
        max_disp_um=max_disp_um,
        max_dt_s=max_dt_s,
    )

    return motion_est
