"""Utility functions for dealing with drifting channels

The main concept here is the "extended geometry" made by the
function `extended_geometry`. The idea is to extend the
probe geometry to cover the range of drift experienced in the
recording. The probe's pitch (unit at which its geometry repeats
vertically) is the integer unit at which we shift channels when
extending the geometry, so that the extended probe contains the
original probe as a subset, as well as copies of the probe shifted
by integer numbers of pitches. As many shifted copies are created
as needed to capture all the drift.
"""
import numpy as np

from .waveform_util import get_pitch


def extended_geometry(geom, motion_est):
    """Extend the probe's channel positions according to the range of motion"""
    assert geom.ndim == 2
    pitch = get_pitch(geom)
    top = geom[:, 1].max()
    bot = geom[:, 1].min()

    # figure out how much upward and downward motion there is
    # recall that z_reg = z - disp_at_s
    max_upward_drift = max(
        0, np.max(-motion_est.disp_at_s(motion_est.time_bin_centers_s, top))
    )
    max_downward_drift = max(
        0, np.max(motion_est.disp_at_s(motion_est.time_bin_centers_s, bot))
    )

    # pad with an integral number of pitches for simplicity
    pitches_pad_up = int(np.ceil(max_upward_drift / pitch))
    pitches_pad_down = int(np.ceil(max_downward_drift / pitch))
    shifted_geoms = [
        geom + [0, pitch * k]
        for k in range(-pitches_pad_down, pitches_pad_up + 1)
    ]

    # all extended site positions
    unique_shifted_positions = np.unique(np.concatenate(shifted_geoms), axis=0)
    # order by depth first, then horizontal position (unique goes the other way)
    extended_geom = unique_shifted_positions[
        np.lexsort(unique_shifted_positions.T)
    ]

    return extended_geom


def occupied_extended_channel_index(times_s, channels, labels, motion_est):
    """Figure out which extended channels each unit appears on"""
    pass


def get_spike_pitch_shifts(times_s, depths_um, geom, motion_est):
    """"""
    pass


def extended_average():
    pass
