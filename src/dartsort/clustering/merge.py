import warnings

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from ..peel.matching_util.pairwise_util import (
    construct_shift_indices,
    iterate_compressed_pairwise_convolutions,
)
from ..templates import TemplateData, template_util
from ..util import job_util
from ..util.data_util import DARTsortSorting
from ..util.internal_config import ComputationConfig, TemplateMergeConfig
from ..util.logging_util import get_logger
from .cluster_util import recluster

logger = get_logger(__name__)


def merge_templates(
    sorting: DARTsortSorting,
    template_data: TemplateData,
    max_shift_samples=40,
    linkage="complete",
    distance_kind="scaled_normeuc",
    sym_function=np.minimum,
    merge_distance_threshold=0.25,
    temporal_upsampling_factor=1,
    amplitude_scaling_variance=0.001,
    amplitude_scaling_boundary=0.1,
    svd_compression_rank=20,
    min_channel_amplitude=0.0,
    spatial_radius_a=None,
    min_spatial_cosine=0.0,
    conv_batch_size=128,
    units_batch_size=8,
    computation_cfg=None,
    show_progress=True,
):
    """Template distance based merge

    Pass in a sorting, recording and template config to make templates,
    and this will merge them. Or, if you have templates
    already, pass them into template_data and we can skip the template
    construction.

    Arguments
    ---------
    max_shift_samples
        Max offset during matching
    amplitude_scaling_*
        Optionally allow scaling during matching

    Returns
    -------
    A new DARTsortSorting
    """
    computation_cfg = job_util.ensure_computation_config(computation_cfg)
    dist_matrix_kwargs = dict(
        sym_function=sym_function,
        max_shift_samples=max_shift_samples,
        temporal_upsampling_factor=temporal_upsampling_factor,
        amplitude_scaling_variance=amplitude_scaling_variance,
        amplitude_scaling_boundary=amplitude_scaling_boundary,
        svd_compression_rank=svd_compression_rank,
        min_channel_amplitude=min_channel_amplitude,
        min_spatial_cosine=min_spatial_cosine,
        conv_batch_size=conv_batch_size,
        units_batch_size=units_batch_size,
        distance_kind=distance_kind,
        spatial_radius_a=spatial_radius_a,
    )
    units, dists, shifts, template_snrs = calculate_merge_distances(
        template_data,
        device=computation_cfg.actual_device(),
        n_jobs=computation_cfg.actual_n_jobs(),
        show_progress=show_progress,
        **dist_matrix_kwargs,  # type: ignore
    )

    # now run hierarchical clustering
    merged_sorting, new_unit_ids = recluster(
        sorting=sorting,
        unit_ids=units,
        dists=dists,
        shifts=shifts.T,
        unit_snrs=template_snrs,
        threshold=merge_distance_threshold,
        link=linkage,
    )

    return dict(sorting=merged_sorting, new_unit_ids=new_unit_ids)


def get_merge_distances(
    template_data: TemplateData,
    template_merge_cfg: TemplateMergeConfig,
    computation_cfg: ComputationConfig | None = None,
    cooccurrence_mask=None,
    show_progress=True,
    conv_batch_size=128,
    units_batch_size=8,
    sampling_frequency: float = 30000.0,
):
    computation_cfg = job_util.ensure_computation_config(computation_cfg)
    shift_samples = int(template_merge_cfg.max_shift_ms * (sampling_frequency / 1000))
    units, dists, shifts, template_snrs = calculate_merge_distances(
        template_data=template_data,
        max_shift_samples=shift_samples,
        temporal_upsampling_factor=template_merge_cfg.temporal_upsampling_factor,
        amplitude_scaling_variance=template_merge_cfg.amplitude_scaling_variance,
        amplitude_scaling_boundary=template_merge_cfg.amplitude_scaling_boundary,
        svd_compression_rank=template_merge_cfg.svd_compression_rank,
        min_spatial_cosine=template_merge_cfg.min_spatial_cosine,
        cooccurrence_mask=cooccurrence_mask,
        conv_batch_size=conv_batch_size,
        units_batch_size=units_batch_size,
        device=computation_cfg.actual_device(),
        n_jobs=computation_cfg.actual_n_jobs(),
        show_progress=show_progress,
        distance_kind=template_merge_cfg.distance_kind,
    )
    return units, dists, shifts, template_snrs


def calculate_merge_distances(
    template_data,
    sym_function=np.minimum,
    max_shift_samples=40,
    temporal_upsampling_factor=1,
    amplitude_scaling_variance=0.001,
    amplitude_scaling_boundary=0.1,
    svd_compression_rank=20,
    min_channel_amplitude=0.0,
    min_spatial_cosine=0.0,
    spatial_radius_a=None,
    cooccurrence_mask=None,
    conv_batch_size=128,
    units_batch_size=8,
    device=None,
    n_jobs=0,
    show_progress=True,
    distance_kind="rms",
):
    # allocate distance + shift matrices. shifts[i,j] is trough[j]-trough[i].
    n_templates = template_data.templates.shape[0]
    sup_dists = np.full((n_templates, n_templates), np.inf)
    sup_shifts = np.zeros((n_templates, n_templates), dtype=int)

    # build distance matrix
    dec_res_iter = get_deconv_resid_decrease_iter(
        template_data,
        max_shift_samples=max_shift_samples,
        temporal_upsampling_factor=temporal_upsampling_factor,
        amplitude_scaling_variance=amplitude_scaling_variance,
        amplitude_scaling_boundary=amplitude_scaling_boundary,
        svd_compression_rank=svd_compression_rank,
        min_channel_amplitude=min_channel_amplitude,
        min_spatial_cosine=min_spatial_cosine,
        cooccurrence_mask=cooccurrence_mask,
        conv_batch_size=conv_batch_size,
        units_batch_size=units_batch_size,
        spatial_radius_a=spatial_radius_a,
        device=device,
        n_jobs=n_jobs,
        show_progress=show_progress,
        distance_kind=distance_kind,
    )
    for res in dec_res_iter:
        if res is None:
            # all pairs in chunk were ignored for one reason or another
            continue

        tixa = res.template_indices_a
        tixb = res.template_indices_b
        sup_dists[tixa, tixb] = res.deconv_resid_decreases / res.template_a_norms  # type: ignore
        sup_shifts[tixa, tixb] = res.shifts  # type: ignore

    units = template_data.unit_ids
    dists = sup_dists
    shifts = sup_shifts
    template_snrs = (
        np.ptp(template_data.templates, 1).max(1) / template_data.spike_counts
    )

    dists = sym_function(dists, dists.T)
    np.fill_diagonal(dists, 0.0)  # sometimes numerical 0 is -1e-6.
    min_dist = dists.min()
    if min_dist < -1e-3:
        warnings.warn(f"Alarmingly negative min distance {min_dist}.")
    dists = np.maximum(dists, 0.0, out=dists)

    return units, dists, shifts, template_snrs


def cross_match_distance_matrix(
    template_data_a,
    template_data_b,
    sym_function=np.minimum,
    max_shift_samples=40,
    temporal_upsampling_factor=1,
    amplitude_scaling_variance=0.001,
    amplitude_scaling_boundary=0.1,
    svd_compression_rank=20,
    min_channel_amplitude=0.0,
    min_spatial_cosine=0.0,
    spatial_radius_a=None,
    conv_batch_size=128,
    units_batch_size=8,
    distance_kind="rms",
    device=None,
    n_jobs=0,
    show_progress=True,
):
    template_data, cross_mask, ids_a, ids_b = combine_templates(
        template_data_a, template_data_b
    )
    units, dists, shifts, template_snrs = calculate_merge_distances(
        template_data,
        sym_function=sym_function,
        max_shift_samples=max_shift_samples,
        temporal_upsampling_factor=temporal_upsampling_factor,
        amplitude_scaling_variance=amplitude_scaling_variance,
        amplitude_scaling_boundary=amplitude_scaling_boundary,
        svd_compression_rank=svd_compression_rank,
        min_channel_amplitude=min_channel_amplitude,
        min_spatial_cosine=min_spatial_cosine,
        spatial_radius_a=spatial_radius_a,
        cooccurrence_mask=cross_mask,
        conv_batch_size=conv_batch_size,
        units_batch_size=units_batch_size,
        distance_kind=distance_kind,
        device=device,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )

    # symmetrize the merge distance matrix, which right now is block off-diagonal
    # (with infs on main diag blocks)
    a_mask = np.flatnonzero(np.isin(units, ids_a))
    b_mask = np.flatnonzero(np.isin(units, ids_b))
    Dab = dists[a_mask[:, None], b_mask[None, :]]
    Dba = dists[b_mask[:, None], a_mask[None, :]]
    Dstack = np.stack((Dab, Dba.T))
    shifts_ab = shifts[a_mask[:, None], b_mask[None, :]]
    shifts_ba = shifts[b_mask[:, None], a_mask[None, :]]
    shifts_stack = np.stack((shifts_ab, -shifts_ba.T))

    # choices[ia, ib] = 0 if Dab[ia, ib] < Dba[ib, ia], else 1
    # but what does that mean? if 0, it means that the upsampled model
    # of B was better able to match A than the other way around. it's not
    # that important, at the end of the day. still, we keep track of it
    # so that we can pick the shift and move on with our lives.
    choices = np.argmin(Dstack, axis=0)
    dists = Dstack[
        choices, np.arange(choices.shape[0])[:, None], np.arange(choices.shape[1])[None]
    ]
    shifts = shifts_stack[
        choices, np.arange(choices.shape[0])[:, None], np.arange(choices.shape[1])[None]
    ]

    # handle duplicates and missing
    a_inds = np.searchsorted(units[a_mask], ids_a, side="right") - 1
    a_kept = units[a_mask][a_inds] == ids_a
    b_inds = np.searchsorted(units[b_mask], ids_b, side="right") - 1
    b_kept = units[b_mask][b_inds] == ids_b
    dists = dists[a_inds[a_kept][:, None], b_inds[b_kept][None, :]]

    snrs_a = template_snrs[a_mask][a_inds[a_kept]]
    snrs_b = template_snrs[b_mask][b_inds[b_kept]]

    return (
        dists,
        shifts,
        snrs_a,
        snrs_b,
        a_kept,
        b_kept,
    )


def get_deconv_resid_decrease_iter(
    template_data,
    max_shift_samples=40,
    temporal_upsampling_factor=8,
    amplitude_scaling_variance=0.001,
    amplitude_scaling_boundary=0.1,
    svd_compression_rank=20,
    min_channel_amplitude=0.0,
    min_spatial_cosine=0.0,
    cooccurrence_mask=None,
    conv_batch_size=128,
    units_batch_size=8,
    spatial_radius_a=None,
    ignore_empty_channels=True,
    distance_kind="rms",
    device=None,
    n_jobs=0,
    show_progress=True,
):
    # get template aux data
    low_rank_templates_b = template_util.svd_compress_templates(
        template_data,
        min_channel_amplitude=min_channel_amplitude,
        rank=svd_compression_rank,
    )
    compressed_upsampled_temporal = template_util.compressed_upsampled_templates(
        low_rank_templates_b.temporal_components,
        ptps=np.ptp(template_data.templates, 1).max(1),
        max_upsample=temporal_upsampling_factor,
    )

    # restrict spatial subset of target templates
    template_data_a = template_data_b = template_data
    low_rank_templates_a = low_rank_templates_b
    if spatial_radius_a:
        template_data_a = template_util.spatially_mask_templates(
            template_data, spatial_radius_a
        )
        low_rank_templates_a = template_util.svd_compress_templates(
            template_data_a,
            min_channel_amplitude=min_channel_amplitude,
            rank=svd_compression_rank,
        )

    # construct helper data and run pairwise convolutions
    (
        template_shift_index_a,
        template_shift_index_b,
        upsampled_shifted_template_index,
        cooccurrence,
    ) = construct_shift_indices(
        chunk_time_centers_s=None,
        template_data_a=template_data,
        compressed_upsampled_temporal=compressed_upsampled_temporal,
        motion=None,
    )
    if cooccurrence_mask is not None:
        cooccurrence = cooccurrence & cooccurrence_mask
    yield from iterate_compressed_pairwise_convolutions(
        template_data_a,
        low_rank_templates_a,
        template_data_b,
        low_rank_templates_b,
        compressed_upsampled_temporal,
        template_shift_index_a,
        template_shift_index_b,
        cooccurrence,
        upsampled_shifted_template_index,
        do_shifting=False,
        reduce_deconv_resid_decrease=True,
        geom=template_data.registered_geom,
        conv_ignore_threshold=0.0,
        min_spatial_cosine=min_spatial_cosine,
        coarse_approx_error_threshold=0.0,
        amplitude_scaling_variance=amplitude_scaling_variance,
        amplitude_scaling_boundary=amplitude_scaling_boundary,
        ignore_empty_channels=ignore_empty_channels,
        distance_kind=distance_kind,
        max_shift=max_shift_samples,  # type: ignore
        conv_batch_size=conv_batch_size,
        units_batch_size=units_batch_size,
        device=device,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )


def combine_templates(template_data_a, template_data_b):
    rgeom = template_data_a.registered_geom
    if rgeom is not None:
        if not np.array_equal(rgeom, template_data_b.registered_geom):
            raise ValueError(
                f"Template data had different registered geoms: "
                f"{template_data_a.registered_geom=} {template_data_b.registered_geom=}"
            )

    ids_a = template_data_a.unit_ids
    ids_b = template_data_b.unit_ids + ids_a.max() + 1
    unit_ids = np.concatenate((ids_a, ids_b))
    templates = np.row_stack((template_data_a.templates, template_data_b.templates))
    spike_counts = np.concatenate(
        (template_data_a.spike_counts, template_data_b.spike_counts)
    )
    scbca = template_data_a.spike_counts_by_channel
    scbcb = template_data_b.spike_counts_by_channel
    if (scbca is not None) and (scbcb is not None):
        spike_counts_by_channel = np.concatenate([scbca, scbcb])
    else:
        spike_counts_by_channel = None

    template_data = TemplateData(
        templates=templates,
        unit_ids=unit_ids,
        spike_counts=spike_counts,
        registered_geom=rgeom,
        spike_counts_by_channel=spike_counts_by_channel,
        trough_offset_samples=template_data_a.trough_offset_samples,
        sampling_frequency=template_data_a.sampling_frequency,
    )

    cross_mask = np.logical_and(
        np.isin(unit_ids, ids_a)[:, None], np.isin(unit_ids, ids_b)[None]
    )
    cross_mask = np.logical_or(cross_mask, cross_mask.T)

    return template_data, cross_mask, ids_a, ids_b
