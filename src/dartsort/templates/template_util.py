import math
from dataclasses import dataclass, replace

import numpy as np
from dartsort.localize.localize_util import localize_waveforms
from dartsort.util import drift_util, waveform_util
from dartsort.util.data_util import DARTsortSorting
from dartsort.util.spiketorch import fast_nanmedian
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


def get_realigned_sorting(
    recording,
    sorting,
    realign_peaks=True,
    low_rank_denoising=False,
    **kwargs,
):
    results = get_templates(
        recording,
        sorting,
        realign_peaks=realign_peaks,
        low_rank_denoising=low_rank_denoising,
        **kwargs,
    )
    return results["sorting"]


def weighted_average(unit_ids, templates, weights):
    n_out = unit_ids.max() + 1
    n_in, t, c = templates.shape
    out = np.zeros((n_out, t, c), dtype=templates.dtype)
    weights = weights.astype(float)
    for i in range(n_out):
        which_in = np.flatnonzero(unit_ids == i)
        if not which_in.size:
            continue

        w = weights[which_in][:, None, None]
        if w.sum() == 0:
            continue
        w /= w.sum()
        out[i] = (w * templates[which_in]).sum(0)

    return out


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
    return_pitch_shifts=False,
    geom_kdtree=None,
    match_distance=None,
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
        target_kdtree=geom_kdtree,
        match_distance=match_distance,
    )
    if return_pitch_shifts:
        return pitch_shifts, unregistered_templates
    return unregistered_templates


def spatially_mask_templates(template_data, radius_um=0.0):
    if not radius_um:
        return template_data

    tt = template_data.templates.copy()
    ci = waveform_util.make_channel_index(template_data.registered_geom, radius_um)
    chans = np.arange(ci.shape[0])
    for j, t in enumerate(tt):
        mask = ~np.isin(chans, ci[np.ptp(t, 0).argmax()])
        tt[j, :, mask] = 0.0

    return replace(template_data, templates=tt)

# -- template numerical processing


@dataclass
class LowRankTemplates:
    temporal_components: np.ndarray
    singular_values: np.ndarray
    spatial_components: np.ndarray
    spike_counts_by_channel: np.ndarray


def svd_compress_templates(
    template_data, min_channel_amplitude=1.0, rank=5, channel_sparse=True
):
    """
    Returns:
    temporal_components: n_units, spike_length_samples, rank
    singular_values: n_units, rank
    spatial_components: n_units, rank, n_channels
    """
    if hasattr(template_data, "templates"):
        templates = template_data.templates
        counts = template_data.spike_counts_by_channel
    else:
        templates = template_data
        counts = None

    vis_mask = np.ptp(templates, axis=1, keepdims=True) > min_channel_amplitude
    vis_templates = templates * vis_mask
    dtype = templates.dtype

    if not channel_sparse:
        U, s, Vh = np.linalg.svd(vis_templates, full_matrices=False)
        # s is descending.
        temporal_components = U[:, :, :rank].astype(dtype)
        singular_values = s[:, :rank].astype(dtype)
        spatial_components = Vh[:, :rank, :].astype(dtype)
        return temporal_components, singular_values, spatial_components

    # channel sparse: only SVD the nonzero channels
    # this encodes the same exact subspace as above, and the reconstruction
    # error is the same as above as a function of rank. it's just that
    # we can zero out some spatial components, which is a useful property
    # (used in pairwise convolutions for instance)
    n, t, c = templates.shape
    temporal_components = np.zeros((n, t, rank), dtype=dtype)
    singular_values = np.zeros((n, rank), dtype=dtype)
    spatial_components = np.zeros((n, rank, c), dtype=dtype)
    for i in range(len(templates)):
        template = templates[i]
        mask = np.flatnonzero(vis_mask[i, 0])
        k = min(rank, mask.size)
        if not k:
            continue
        U, s, Vh = np.linalg.svd(template[:, mask], full_matrices=False)
        temporal_components[i, :, :k] = U[:, :rank]
        singular_values[i, :k] = s[:rank]
        spatial_components[i, :k, mask] = Vh[:rank].T

    return LowRankTemplates(temporal_components, singular_values, spatial_components, counts)


def temporally_upsample_templates(
    templates, temporal_upsampling_factor=8, kind="cubic"
):
    """Note, also works on temporal components thanks to compatible shape."""
    n, t, c = templates.shape
    tp = np.arange(t).astype(float)
    erp = interp1d(tp, templates, axis=1, bounds_error=True, kind=kind)
    tup = np.arange(t, step=1.0 / temporal_upsampling_factor)
    tup.clip(0, t - 1, out=tup)
    upsampled_templates = erp(tup)
    upsampled_templates = upsampled_templates.reshape(
        n, t, temporal_upsampling_factor, c
    )
    upsampled_templates = upsampled_templates.astype(templates.dtype)
    return upsampled_templates


@dataclass
class CompressedUpsampledTemplates:
    n_compressed_upsampled_templates: int
    compressed_upsampled_templates: np.ndarray
    compressed_upsampling_map: np.ndarray
    compressed_upsampling_index: np.ndarray
    compressed_index_to_template_index: np.ndarray
    compressed_index_to_upsampling_index: np.ndarray


def default_n_upsamples_map(ptps, max_upsample=8):
    # avoid overflow in 4 ** by trimming ptp range in advance
    max_ptp = 1 + 2 * math.log(max_upsample, 4)
    ptps = np.minimum(ptps, max_ptp)
    upsamples = 4 ** (ptps // 2)
    return np.clip(upsamples, 1, max_upsample).astype(int)


def compressed_upsampled_templates(
    templates,
    ptps=None,
    max_upsample=8,
    n_upsamples_map=default_n_upsamples_map,
    kind="cubic",
):
    """compressedly store fewer temporally upsampled copies of lower amplitude templates

    Returns
    -------
    A CompressedUpsampledTemplates object with fields:
        compressed_upsampled_templates : array (n_compressed_upsampled_templates, spike_length_samples, n_channels)
        compressed_upsampling_map : array (n_templates, max_upsample)
            compressed_upsampled_templates[compressed_upsampling_map[unit, j]] is an approximation
            of the jth upsampled template for this unit. for low-amplitude units,
            compressed_upsampling_map[unit] will have fewer unique entries, corresponding
            to fewer saved upsampled copies for that unit.
        compressed_upsampling_index : array (n_templates, max_upsample)
            A n_compressed_upsampled_templates-padded ragged array mapping each
            template index to its compressed upsampled indices
        compressed_index_to_template_index
        compressed_index_to_upsampling_index
    """
    n_templates = templates.shape[0]
    if max_upsample == 1:
        return CompressedUpsampledTemplates(
            n_templates,
            templates,
            np.arange(n_templates)[:, None],
            np.arange(n_templates)[:, None],
            np.arange(n_templates),
            np.zeros(n_templates, dtype=int),
        )

    # how many copies should each unit get?
    # sometimes users may pass temporal SVD components in instead of templates,
    # so we allow them to pass in the amplitudes of the actual templates
    if ptps is None:
        ptps = np.ptp(templates, 1).max(1)
    assert ptps.shape == (n_templates,)
    if n_upsamples_map is None:
        n_upsamples = np.full(n_templates, max_upsample)
    else:
        n_upsamples = n_upsamples_map(ptps, max_upsample=max_upsample)

    # build the compressed upsampling map
    compressed_upsampling_map = np.full((n_templates, max_upsample), -1, dtype=int)
    compressed_upsampling_index = np.full((n_templates, max_upsample), -1, dtype=int)
    template_indices = []
    upsampling_indices = []
    current_compressed_index = 0
    for i, nup in enumerate(n_upsamples):
        compression = max_upsample // nup
        nup = max_upsample // compression  # handle divisibility failure

        # new compressed indices
        compressed_upsampling_map[i] = current_compressed_index + np.arange(nup).repeat(
            compression
        )
        compressed_upsampling_index[i, :nup] = current_compressed_index + np.arange(nup)
        current_compressed_index += nup

        # indices of the templates to keep in the full array of upsampled templates
        template_indices.extend([i] * nup)
        upsampling_indices.extend(compression * np.arange(nup))
    assert (compressed_upsampling_map >= 0).all()
    assert (
        np.unique(compressed_upsampling_map).size
        == (compressed_upsampling_index >= 0).sum()
        == compressed_upsampling_map.max() + 1
        == compressed_upsampling_index.max() + 1
        == current_compressed_index
    )
    template_indices = np.array(template_indices)
    upsampling_indices = np.array(upsampling_indices)
    compressed_upsampling_index[
        compressed_upsampling_index < 0
    ] = current_compressed_index

    # get the upsampled templates
    all_upsampled_templates = temporally_upsample_templates(
        templates, temporal_upsampling_factor=max_upsample, kind=kind
    )
    # n, up, t, c
    all_upsampled_templates = all_upsampled_templates.transpose(0, 2, 1, 3)
    rix = np.ravel_multi_index(
        (template_indices, upsampling_indices), all_upsampled_templates.shape[:2]
    )
    all_upsampled_templates = all_upsampled_templates.reshape(
        n_templates * max_upsample, templates.shape[1], templates.shape[2]
    )
    compressed_upsampled_templates = all_upsampled_templates[rix]

    return CompressedUpsampledTemplates(
        current_compressed_index,
        compressed_upsampled_templates,
        compressed_upsampling_map,
        compressed_upsampling_index,
        template_indices,
        upsampling_indices,
    )
