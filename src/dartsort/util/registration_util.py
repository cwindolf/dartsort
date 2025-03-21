import pickle
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
    output_directory=None,
    overwrite=False,
    do_motion_estimation=True,
    probe_boundary_padding_um=100.0,
    spatial_bin_length_um: float = 1.0,
    temporal_bin_length_s: float = 1.0,
    window_step_um: float = 400.0,
    window_scale_um: float = 450.0,
    window_margin_um: Optional[float] = None,
    max_dt_s: float = 1000.0,
    max_disp_um: Optional[float] = None,
    correlation_threshold: float = 0.1,
    min_amplitude: Optional[float] = None,
    masked_correlation: bool = False,
    weights_threshold: float = 0.2,
    rigid: bool=False,
    localizations_dataset_name="point_source_localizations",
    amplitudes_dataset_name="denoised_ptp_amplitudes",
    device: Optional["torch.device"]=None,
):
    if not do_motion_estimation:
        return None

    if output_directory is not None:
        motion_est_pkl = output_directory / "motion_est.pkl"
        if not overwrite and motion_est_pkl.exists():
            with open(motion_est_pkl, "rb") as jar:
                return pickle.load(jar)

    if not have_dredge:
        raise ValueError("Please install DREDge to use motion estimation.")

    x = getattr(sorting, localizations_dataset_name)[:, 0]
    z = getattr(sorting, localizations_dataset_name)[:, 2]
    a = getattr(sorting, amplitudes_dataset_name)
    geom = recording.get_channel_locations()
    xmin = geom[:, 0].min() - probe_boundary_padding_um
    xmax = geom[:, 0].max() + probe_boundary_padding_um
    zmin = geom[:, 1].min() - probe_boundary_padding_um
    zmax = geom[:, 1].max() + probe_boundary_padding_um
    valid = x == x.clip(xmin, xmax)
    valid &= z == z.clip(zmin, zmax)
    if min_amplitude:
        valid &= a >= min_amplitude
    valid = np.flatnonzero(valid)

    # features for registration
    z = z[valid]
    t_s = sorting.times_seconds[valid]
    a = a[valid]

    # run registration
    motion_est, info = dredge_ap.register(
        amps=a,
        depths_um=z,
        times_s=t_s,
        rigid=rigid,
        bin_um=spatial_bin_length_um,
        bin_s=temporal_bin_length_s,
        win_step_um=window_step_um,
        weights_threshold_low=weights_threshold,
        weights_threshold_high=weights_threshold,
        win_scale_um=window_scale_um,
        win_margin_um=window_margin_um,
        count_masked_correlation=masked_correlation,
        max_disp_um=max_disp_um,
        max_dt_s=max_dt_s,
        mincorr=correlation_threshold,
        device=device,
    )

    if output_directory is not None:
        with open(motion_est_pkl, "wb") as jar:
            pickle.dump(motion_est, jar)

    return motion_est
