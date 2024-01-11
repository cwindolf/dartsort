"""Computing relocated waveforms used in clustering

These waveforms are first shifted by an integer number of
probe pitches to account for gross drift and then rescaled
on each channel according to the point source model to
account for sub-pitch motion.

This is in numpy, since it's part of the clustering which
is all just numpy, and that's why it lives here and not
in transform/ which has torch-based stuff for use during
peeling.
"""
import numpy as np
from dartsort.util import drift_util


def relocated_waveforms_on_static_channels(
    waveforms,
    main_channels,
    channel_index,
    target_channels,
    xyza_from,
    z_to,
    geom,
    registered_geom=None,
    target_kdtree=None,
    match_distance=None,
    fill_value=np.nan,
):
    """Compute relocated waveforms"""
    two_d = waveforms.ndim == 2
    if two_d:
        waveforms = waveforms[:, None, :]
    x, y, z_from, alpha = xyza_from.T
    if registered_geom is None:
        registered_geom = geom

    # -- handle coarse drift (larger than a pitch)
    n_pitches_shift = drift_util.get_spike_pitch_shifts(
        z_from, geom, registered_depths_um=z_to
    )
    shifted_waveforms = drift_util.get_waveforms_on_static_channels(
        waveforms,
        main_channels=main_channels,
        channel_index=channel_index,
        target_channels=target_channels,
        n_pitches_shift=n_pitches_shift,
        geom=geom,
        registered_geom=registered_geom,
        target_kdtree=target_kdtree,
        match_distance=match_distance,
        fill_value=fill_value,
    )

    # -- handle sub-pitch drift
    # account for the already applied coarse drift correction
    pitch = drift_util.get_pitch(geom)
    z_from = z_from + pitch * n_pitches_shift
    original_amplitudes = point_source_amplitude_vectors(
        x, y, z_from, alpha, registered_geom, channels=target_channels
    )
    target_amplitudes = point_source_amplitude_vectors(
        x, y, z_to, alpha, registered_geom, channels=target_channels
    )
    rescaling = target_amplitudes / original_amplitudes
    shifted_waveforms *= rescaling[:, None, :]
    
    if two_d:
        shifted_waveforms = shifted_waveforms[:, 0, :]

    return shifted_waveforms


def point_source_amplitude_vectors(x, y, z_abs, alpha, geom, channels=None):
    """Point-source amplitudes on fixed channels from varying spike positions"""
    if channels is not None:
        geom = geom[channels]

    dx = geom[None, :, 0] - x[:, None]
    dz = geom[None, :, 1] - z_abs[:, None]
    dy = y[:, None]

    return alpha[:, None] / np.sqrt(dy**2 + dx**2 + dz**2)
