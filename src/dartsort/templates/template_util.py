import numpy as np
from dartsort.localize.localize_util import localize_waveforms
from dartsort.util import drift_util
from dartsort.util.data_util import DARTsortSorting
from dartsort.util.waveform_util import fast_nanmedian
from scipy.interpolate import interp1d

from .get_templates import get_raw_templates, get_templates

# -- alternate template constructors


def get_single_raw_template(
    recording,
    spike_times_samples,
    trough_offset_samples=42,
    spike_length_samples=121,
    spikes_per_unit=500,
    reducer=np.median,
    random_seed=0,
):
    single_sorting = DARTsortSorting(
        spike_times_samples, channels=np.zeros_like(spike_times_samples)
    )
    return get_raw_templates(
        recording,
        single_sorting,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        spikes_per_unit=spikes_per_unit,
        reducer=reducer,
        random_seed=random_seed,
        realign_peaks=False,
        show_progress=False,
    )


def get_registered_templates(
    recording,
    sorting,
    spike_times_s,
    spike_depths_um,
    geom,
    motion_est,
    registered_template_depths_um=None,
    localization_radius_um=100,
    trough_offset_samples=42,
    spike_length_samples=121,
    spikes_per_unit=500,
    realign_peaks=False,
    realign_max_sample_shift=20,
    low_rank_denoising=True,
    denoising_tsvd=None,
    denoising_rank=5,
    denoising_fit_radius=75,
    denoising_spikes_fit=50_000,
    denoising_snr_threshold=50.0,
    zero_radius_um=None,
    reducer=fast_nanmedian,
    random_seed=0,
    n_jobs=0,
    show_progress=True,
):
    # use geometry and motion estimate to get pitch shifts and reg geom
    registered_geom = drift_util.registered_geometry(geom, motion_est=motion_est)
    pitch_shifts = drift_util.get_spike_pitch_shifts(
        spike_depths_um, geom, times_s=spike_times_s, motion_est=motion_est
    )

    # now compute templates
    results = get_templates(
        recording,
        sorting,
        registered_geom=registered_geom,
        pitch_shifts=pitch_shifts,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        spikes_per_unit=spikes_per_unit,
        realign_peaks=realign_peaks,
        realign_max_sample_shift=realign_max_sample_shift,
        low_rank_denoising=low_rank_denoising,
        denoising_tsvd=denoising_tsvd,
        denoising_rank=denoising_rank,
        denoising_fit_radius=denoising_fit_radius,
        denoising_spikes_fit=denoising_spikes_fit,
        denoising_snr_threshold=denoising_snr_threshold,
        zero_radius_um=zero_radius_um,
        reducer=reducer,
        random_seed=random_seed,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )

    # and, localize them since users of these templates will want locations
    registered_template_depths_um = get_template_depths(
        results["templates"],
        registered_geom,
        localization_radius_um=localization_radius_um,
    )

    results["registered_template_depths_um"] = registered_template_depths_um
    results["registered_geom"] = registered_geom
    results["registered_templates"] = results["templates"]

    return results


# -- template drift handling


def get_template_depths(templates, geom, localization_radius_um=100):
    template_locs = localize_waveforms(
        templates, geom=geom, radius=localization_radius_um
    )
    template_depths_um = template_locs["z_abs"]
    return template_depths_um


def templates_at_time(
    t_s,
    registered_templates,
    geom,
    registered_template_depths_um=None,
    registered_geom=None,
    motion_est=None,
):
    if registered_geom is None:
        return registered_templates
    assert motion_est is not None
    assert registered_template_depths_um is not None

    # for each unit, extract relevant channels at time t_s
    # how many pitches to shift each unit relative to registered pos at time t_s?
    unregistered_depths_um = drift_util.invert_motion_estimate(
        motion_est, t_s, registered_template_depths_um
    )
    # reverse arguments to pitch shifts since we are going the other direction
    pitch_shifts = drift_util.get_spike_pitch_shifts(
        depths_um=registered_template_depths_um,
        geom=geom,
        registered_depths_um=unregistered_depths_um,
    )
    # extract relevant channel neighborhoods, also by reversing args to a drift helper
    unregistered_templates = drift_util.get_waveforms_on_static_channels(
        registered_templates,
        registered_geom,
        n_pitches_shift=pitch_shifts,
        registered_geom=geom,
        fill_value=np.nan,
    )
    assert not np.isnan(unregistered_templates).any()

    return unregistered_templates


# -- template numerical processing


def svd_compress_templates(templates, min_channel_amplitude=1.0, rank=5):
    """
    Returns:
    temporal_components: n_units, spike_length_samples, rank
    singular_values: n_units, rank
    spatial_components: n_units, rank, n_channels
    """
    vis_mask = templates.ptp(axis=1, keepdims=True) > min_channel_amplitude
    vis_templates = templates * vis_mask
    U, s, Vh = np.linalg.svd(vis_templates, full_matrices=False)
    # s is descending.
    temporal_components = U[..., :, :rank]
    singular_values = s[..., :rank]
    spatial_components = Vh[..., :rank, :]
    return temporal_components, singular_values, spatial_components


def temporally_upsample_templates(
    templates, temporal_upsampling_factor=8, kind="cubic"
):
    """Note, also works on temporal components thanks to compatible shape."""
    n, t, c = templates.shape
    tp = np.arange(t).astype(float)
    erp = interp1d(tp, templates, axis=1, bounds_error=True)
    tup = np.arange(t, step=1. / temporal_upsampling_factor)
    upsampled_templates = erp(tup)
    upsampled_templates = upsampled_templates.reshape(n, t, temporal_upsampling_factor, c)
    return upsampled_templates
