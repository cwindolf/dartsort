from dataclasses import replace
from typing import Optional

import numpy as np
from dartsort.config import TemplateConfig
from dartsort.templates import TemplateData, template_util
from dartsort.templates.pairwise_util import (
    construct_shift_indices, iterate_compressed_pairwise_convolutions)
from dartsort.util.data_util import DARTsortSorting
from scipy.cluster.hierarchy import complete, fcluster
from scipy.sparse import coo_array
from scipy.sparse.csgraph import maximum_bipartite_matching
from tqdm.auto import tqdm


def merge_templates(
    sorting: DARTsortSorting,
    recording,
    template_data: Optional[TemplateData] = None,
    template_config: Optional[TemplateConfig] = None,
    motion_est=None,
    max_shift_samples=20,
    superres_linkage=np.max,
    sym_function=np.minimum,
    merge_distance_threshold=0.25,
    temporal_upsampling_factor=8,
    amplitude_scaling_variance=0.0,
    amplitude_scaling_boundary=0.5,
    svd_compression_rank=10,
    min_channel_amplitude=0.0,
    min_spatial_cosine=0.0,
    conv_batch_size=128,
    units_batch_size=8,
    device=None,
    n_jobs=0,
    n_jobs_templates=0,
    template_save_folder=None,
    overwrite_templates=False,
    show_progress=True,
    template_npz_filename="template_data.npz",
) -> DARTsortSorting:
    """Template distance based merge

    Pass in a sorting, recording and template config to make templates,
    and this will merge them (with superres). Or, if you have templates
    already, pass them into template_data and we can skip the template
    construction.

    Arguments
    ---------
    max_shift_samples
        Max offset during matching
    superres_linkage
        How to combine distances between two units' superres templates
        By default, it's the max.
    amplitude_scaling_*
        Optionally allow scaling during matching

    Returns
    -------
    A new DARTsortSorting
    """
    if template_data is None:
        template_data = TemplateData.from_config(
            recording,
            sorting,
            template_config,
            motion_est=motion_est,
            n_jobs=n_jobs_templates,
            save_folder=template_save_folder,
            overwrite=overwrite_templates,
            device=device,
            save_npz_name=template_npz_filename,
        )

    units, dists, shifts, template_snrs = calculate_merge_distances(
        template_data,
        superres_linkage=superres_linkage,
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
        device=device,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )

    # now run hierarchical clustering
    return recluster(
        sorting,
        units,
        dists,
        shifts,
        template_snrs,
        merge_distance_threshold=merge_distance_threshold,
    )


def merge_across_sortings(
    sortings,
    recording,
    template_config: Optional[TemplateConfig] = None,
    motion_est=None,
    cross_merge_distance_threshold=0.5,
    within_merge_distance_threshold=0.5,
    superres_linkage=np.max,
    sym_function=np.minimum,
    max_shift_samples=20,
    temporal_upsampling_factor=8,
    amplitude_scaling_variance=0.0,
    amplitude_scaling_boundary=0.5,
    svd_compression_rank=10,
    min_channel_amplitude=0.0,
    min_spatial_cosine=0.0,
    conv_batch_size=128,
    units_batch_size=8,
    device=None,
    n_jobs=0,
    n_jobs_templates=0,
    show_progress=True,
):
    # first, merge within chunks
    if within_merge_distance_threshold:
        sortings = [
            merge_templates(
                sorting,
                recording,
                template_config=template_config,
                motion_est=motion_est,
                max_shift_samples=max_shift_samples,
                superres_linkage=superres_linkage,
                sym_function=sym_function,
                merge_distance_threshold=within_merge_distance_threshold,
                temporal_upsampling_factor=temporal_upsampling_factor,
                amplitude_scaling_variance=amplitude_scaling_variance,
                amplitude_scaling_boundary=amplitude_scaling_boundary,
                min_spatial_cosine=min_spatial_cosine,
                svd_compression_rank=svd_compression_rank,
                min_channel_amplitude=min_channel_amplitude,
                conv_batch_size=conv_batch_size,
                units_batch_size=units_batch_size,
                device=device,
                n_jobs=n_jobs,
                n_jobs_templates=n_jobs_templates,
                show_progress=False,
            )
            for sorting in tqdm(sortings, desc="Merge within chunks")
        ]

    # now, across chunks
    for i in range(len(sortings) - 1):
        template_data_a = TemplateData.from_config(
            recording,
            sortings[i],
            template_config,
            motion_est=motion_est,
            n_jobs=n_jobs_templates,
            device=device,
        )
        template_data_b = TemplateData.from_config(
            recording,
            sortings[i + 1],
            template_config,
            motion_est=motion_est,
            n_jobs=n_jobs_templates,
            device=device,
        )
        dists, shifts, snrs_a, snrs_b, units_a, units_b = cross_match_distance_matrix(
            template_data_a,
            template_data_b,
            superres_linkage=superres_linkage,
            sym_function=sym_function,
            max_shift_samples=max_shift_samples,
            temporal_upsampling_factor=temporal_upsampling_factor,
            amplitude_scaling_variance=amplitude_scaling_variance,
            amplitude_scaling_boundary=amplitude_scaling_boundary,
            min_spatial_cosine=min_spatial_cosine,
            svd_compression_rank=svd_compression_rank,
            min_channel_amplitude=min_channel_amplitude,
            conv_batch_size=conv_batch_size,
            units_batch_size=units_batch_size,
            device=device,
            n_jobs=n_jobs,
            show_progress=show_progress,
        )
        sortings[i], sortings[i + 1] = cross_match(
            sortings[i],
            sortings[i + 1],
            dists,
            shifts,
            snrs_a,
            snrs_b,
            units_a,
            units_b,
            merge_distance_threshold=cross_merge_distance_threshold,
        )

    # combine into one big sorting
    return combine_sortings(sortings)


def calculate_merge_distances(
    template_data,
    superres_linkage=np.max,
    sym_function=np.minimum,
    max_shift_samples=20,
    temporal_upsampling_factor=8,
    amplitude_scaling_variance=0.0,
    amplitude_scaling_boundary=0.5,
    svd_compression_rank=10,
    min_channel_amplitude=0.0,
    min_spatial_cosine=0.0,
    cooccurrence_mask=None,
    conv_batch_size=128,
    units_batch_size=8,
    device=None,
    n_jobs=0,
    show_progress=True,
):
    # allocate distance + shift matrices. shifts[i,j] is trough[j]-trough[i].
    n_templates = template_data.templates.shape[0]
    sup_dists = np.full((n_templates, n_templates), np.inf)
    sup_shifts = np.zeros((n_templates, n_templates), dtype=int)

    # build distance matrix
    dec_res_iter = get_deconv_resid_norm_iter(
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
        device=device,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )
    for res in dec_res_iter:
        tixa = res.template_indices_a
        tixb = res.template_indices_b
        rms_ratio = res.deconv_resid_norms / res.template_a_norms
        sup_dists[tixa, tixb] = rms_ratio
        sup_shifts[tixa, tixb] = res.shifts

    # apply linkage to reduce across superres templates
    units = np.unique(template_data.unit_ids)
    if units.size < n_templates:
        dists = np.full((units.size, units.size), np.inf)
        shifts = np.zeros((units.size, units.size), dtype=int)
        for ia, ua in enumerate(units):
            in_ua = np.flatnonzero(template_data.unit_ids == ua)
            for ib, ub in enumerate(units):
                in_ub = np.flatnonzero(template_data.unit_ids == ub)
                in_pair = (in_ua[:, None], in_ub[None, :])
                dists[ia, ib] = superres_linkage(sup_dists[in_pair])
                shifts[ia, ib] = np.median(sup_shifts[in_pair])
        coarse_td = template_data.coarsen(with_locs=False)
        template_snrs = coarse_td.templates.ptp(1).max(1) / coarse_td.spike_counts
    else:
        dists = sup_dists
        shifts = sup_shifts
        template_snrs = (
            template_data.templates.ptp(1).max(1) / template_data.spike_counts
        )

    dists = sym_function(dists, dists.T)

    return units, dists, shifts, template_snrs


def cross_match_distance_matrix(
    template_data_a,
    template_data_b,
    superres_linkage=np.max,
    sym_function=np.minimum,
    max_shift_samples=20,
    temporal_upsampling_factor=8,
    amplitude_scaling_variance=0.0,
    amplitude_scaling_boundary=0.5,
    svd_compression_rank=10,
    min_channel_amplitude=0.0,
    min_spatial_cosine=0.0,
    conv_batch_size=128,
    units_batch_size=8,
    device=None,
    n_jobs=0,
    show_progress=False,
):
    template_data, cross_mask, ids_a, ids_b = combine_templates(
        template_data_a, template_data_b
    )
    units, dists, shifts, template_snrs = calculate_merge_distances(
        template_data,
        superres_linkage=superres_linkage,
        sym_function=sym_function,
        max_shift_samples=max_shift_samples,
        temporal_upsampling_factor=temporal_upsampling_factor,
        amplitude_scaling_variance=amplitude_scaling_variance,
        amplitude_scaling_boundary=amplitude_scaling_boundary,
        svd_compression_rank=svd_compression_rank,
        min_channel_amplitude=min_channel_amplitude,
        min_spatial_cosine=min_spatial_cosine,
        cooccurrence_mask=cross_mask,
        conv_batch_size=conv_batch_size,
        units_batch_size=units_batch_size,
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
    dists = Dstack[choices]
    shifts = shifts_stack[choices]

    snrs_a = template_snrs[a_mask]
    snrs_b = template_snrs[b_mask]
    units_a = units[a_mask]
    units_b = units[b_mask]

    return dists, shifts, snrs_a, snrs_b, units_a, units_b


def recluster(
    sorting,
    units,
    dists,
    shifts,
    template_snrs,
    merge_distance_threshold=0.25,
):

    # upper triangle not including diagonal, aka condensed distance matrix in scipy
    pdist = dists[np.triu_indices(dists.shape[0], k=1)]
    # scipy hierarchical clustering only supports finite values, so let's just
    # drop in a huge value here
    pdist[~np.isfinite(pdist)] = 1_000_000 + pdist[np.isfinite(pdist)].max()
    # complete linkage: max dist between all pairs across clusters.
    Z = complete(pdist)
    # extract flat clustering using our max dist threshold
    new_labels = fcluster(Z, merge_distance_threshold, criterion="distance")

    # update labels
    labels_updated = sorting.labels.copy()
    kept = np.flatnonzero(np.isin(sorting.labels, units))
    _, flat_labels = np.unique(labels_updated[kept], return_inverse=True)
    labels_updated[kept] = new_labels[flat_labels]

    # update times according to shifts
    times_updated = sorting.times_samples.copy()

    # find original labels in each cluster
    clust_inverse = {i: [] for i in new_labels}
    for orig_label, new_label in enumerate(new_labels):
        clust_inverse[new_label].append(orig_label)

    # align to best snr unit
    for new_label, orig_labels in clust_inverse.items():
        # we don't need to realign clusters which didn't change
        if len(orig_labels) <= 1:
            continue

        orig_snrs = template_snrs[orig_labels]
        best_orig = orig_labels[orig_snrs.argmax()]
        for ogl in np.setdiff1d(orig_labels, [best_orig]):
            in_orig_unit = np.flatnonzero(sorting.labels == ogl)
            # this is like trough[best] - trough[ogl]
            shift_og_best = shifts[ogl, best_orig]
            # if >0, trough of og is behind trough of best.
            # subtracting will move trough of og to the right.
            times_updated[in_orig_unit] -= shift_og_best

    return replace(sorting, times_samples=times_updated, labels=labels_updated)


def cross_match(
    sorting_a,
    sorting_b,
    dists,
    shifts,
    snrs_a,
    snrs_b,
    units_a,
    units_b,
    merge_distance_threshold=0.5,
):
    assert np.array_equal(units_a, sorting_a.units)
    assert np.array_equal(units_b, sorting_b.units)

    ia, ib = np.nonzero(dists <= merge_distance_threshold)
    weights = coo_array((-dists[ia, ib], (ia.astype(np.intc), ib.astype(np.intc))))
    b_to_a = maximum_bipartite_matching(weights)
    assert b_to_a.shape == units_b.shape

    # -- update sortings
    # sorting A's labels don't change
    # matched B units are given their A-match's label, and unmatched units get labels
    # starting from A's next cluster label
    matched = b_to_a >= 0
    next_a_label = units_a.max() + 1
    b_reindex = np.full_like(units_b, -1)
    matched_a_units = units_a[b_to_a[matched]]
    b_reindex[matched] = matched_a_units
    b_reindex[~matched] = next_a_label + np.arange(np.count_nonzero(~matched))
    b_labels = b_reindex[sorting_b.labels]

    # both sortings' times can change. we shift the lower SNR unit.
    # shifts is like trough[a] - trough[b]. if >0, subtract from a or add to b to realign.
    matched_b_units = units_b[matched]
    shifts = shifts[matched_a_units, matched_b_units]

    shifts_a = np.zeros_like(sorting_a.times_samples)
    a_matched = np.flatnonzero(np.isin(sorting_a.labels, matched_a_units))
    a_match_ix = np.searchsorted(matched_a_units, sorting_a.labels[a_matched])
    shifts_a[a_matched] = shifts[a_match_ix]
    times_a = sorting_a.times_samples - shifts_a

    shifts_b = np.zeros_like(sorting_b.times_samples)
    b_matched = np.flatnonzero(np.isin(sorting_b.labels, matched_b_units))
    b_match_ix = np.searchsorted(matched_b_units, sorting_b.labels[b_matched])
    shifts_b[b_matched] = shifts[b_match_ix]
    times_b = sorting_b.times_samples - shifts_b

    sorting_a = replace(sorting_a, times_samples=times_a)
    sorting_b = replace(sorting_b, labels=b_labels, times_samples=times_b)
    return sorting_a, sorting_b


def get_deconv_resid_norm_iter(
    template_data,
    max_shift_samples=20,
    temporal_upsampling_factor=8,
    amplitude_scaling_variance=0.0,
    amplitude_scaling_boundary=0.5,
    svd_compression_rank=10,
    min_channel_amplitude=0.0,
    min_spatial_cosine=0.0,
    cooccurrence_mask=None,
    conv_batch_size=128,
    units_batch_size=8,
    device=None,
    n_jobs=0,
    show_progress=True,
):
    # get template aux data
    low_rank_templates = template_util.svd_compress_templates(
        template_data.templates,
        min_channel_amplitude=min_channel_amplitude,
        rank=svd_compression_rank,
    )
    compressed_upsampled_temporal = template_util.compressed_upsampled_templates(
        low_rank_templates.temporal_components,
        ptps=template_data.templates.ptp(1).max(1),
        max_upsample=temporal_upsampling_factor,
    )

    # construct helper data and run pairwise convolutions
    (
        template_shift_index_a,
        template_shift_index_b,
        upsampled_shifted_template_index,
        cooccurrence,
    ) = construct_shift_indices(
        None,
        None,
        template_data,
        compressed_upsampled_temporal,
        motion_est=None,
    )
    if cooccurrence_mask is not None:
        cooccurrence = cooccurrence & cooccurrence_mask
    yield from iterate_compressed_pairwise_convolutions(
        template_data,
        low_rank_templates,
        template_data,
        low_rank_templates,
        compressed_upsampled_temporal,
        template_shift_index_a,
        template_shift_index_b,
        cooccurrence,
        upsampled_shifted_template_index,
        do_shifting=False,
        reduce_deconv_resid_norm=True,
        geom=template_data.registered_geom,
        conv_ignore_threshold=0.0,
        min_spatial_cosine=min_spatial_cosine,
        coarse_approx_error_threshold=0.0,
        amplitude_scaling_variance=amplitude_scaling_variance,
        amplitude_scaling_boundary=amplitude_scaling_boundary,
        max_shift=max_shift_samples,
        conv_batch_size=conv_batch_size,
        units_batch_size=units_batch_size,
        device=device,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )


def combine_templates(template_data_a, template_data_b):
    rgeom = template_data_a.registered_geom
    if rgeom is not None:
        assert np.array_equal(rgeom, template_data_b.registered_geom)
    locs = template_data_a.registered_template_depths_um
    if locs is not None:
        locs = np.concatenate((locs, template_data_b.registered_template_depths_um))

    ids_a = template_data_a.unit_ids
    ids_b = template_data_b.unit_ids + ids_a.max() + 1
    unit_ids = np.concatenate((ids_a, ids_b))
    templates = np.row_stack((template_data_a.templates, template_data_b.templates))
    spike_counts = np.concatenate(
        (template_data_a.spike_counts, template_data_b.spike_counts)
    )
    template_data = TemplateData(
        templates=templates,
        unit_ids=unit_ids,
        spike_counts=spike_counts,
        registered_geom=rgeom,
        registered_template_depths_um=locs,
    )

    cross_mask = np.zeros((unit_ids.size, unit_ids.size), dtype=bool)
    cross_mask[ids_a.size:, :ids_b.size] = True
    cross_mask[:ids_a.size, ids_b.size:] = True

    return template_data, cross_mask, ids_a, ids_b


def combine_sortings(sortings):
    labels = np.full_like(sortings[0].labels, -1)
    times_samples = sortings[0].times_samples.copy()

    for sorting in sortings:
        kept = np.flatnonzero(sorting.labels >= 0)
        assert np.all(labels[kept] < 0)
        labels[kept] = sorting.labels[kept]
        times_samples[kept] = sorting.times_samples[kept]

    sorting = replace(sortings[0], labels=labels, times_samples=times_samples)
    return sorting
