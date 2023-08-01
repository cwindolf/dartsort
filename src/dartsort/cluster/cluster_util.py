import numpy as np
from dartsort.util import drift_util
from dredge.motion_util import IdentityMotionEstimate
from scipy.spatial import KDTree


def closest_registered_channels(
    times_seconds, x, z_abs, geom, motion_est=None
):
    """Assign spikes to the drift-extended channel closest to their registered position"""
    if motion_est is None:
        motion_est == IdentityMotionEstimate()
    registered_geom = drift_util.registered_geometry(geom, motion_est)
    z_reg = motion_est.correct_s(times_seconds, z_abs)
    reg_pos = np.c_[x, z_reg]

    registered_kdt = KDTree(registered_geom)
    distances, reg_channels = registered_kdt.query(reg_pos)

    return reg_channels
