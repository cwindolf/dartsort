from __future__ import annotations  # allow forward type references

from collections import namedtuple
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterator, Optional, Union

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from dartsort.util import drift_util
from dartsort.util.multiprocessing_util import get_pool
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from tqdm.auto import tqdm

from . import template_util, templates


def compressed_convolve_to_h5(
    output_hdf5_filename,
    template_data: templates.TemplateData,
    low_rank_templates: template_util.LowRankTemplates,
    compressed_upsampled_temporal: template_util.CompressedUpsampledTemplates,
    chunk_time_centers_s: Optional[np.ndarray] = None,
    motion_est=None,
    geom: Optional[np.ndarray] = None,
    conv_ignore_threshold=0.0,
    coarse_approx_error_threshold=0.0,
    conv_batch_size=1024,
    units_batch_size=8,
    overwrite=False,
    device=None,
    n_jobs=0,
    show_progress=True,
):
    """Convolve all pairs of templates and store result in a .h5

    See pairwise.CompressedPairwiseConvDB for how to read the
    resulting convolutions back.

    This runs compressed_convolve_pairs in a loop over chunks
    of unit pairs, so that it's not all done in memory at one time,
    and so that it can be done in parallel.
    """
    if overwrite:
        pass  # TODO

    # construct indexing helpers
    template_shift_index = drift_util.get_shift_and_unit_pairs(
        chunk_time_centers_s,
        geom,
        template_data,
        motion_est=motion_est,
    )
    upsampled_shifted_template_index = get_upsampled_shifted_template_index(
        template_shift_index, compressed_upsampled_temporal
    )
    print(f"compressed_convolve_to_h5 {conv_batch_size=} {units_batch_size=} {device=}")

    chunk_res_iterator = iterate_compressed_pairwise_convolutions(
        template_data=template_data,
        low_rank_templates=low_rank_templates,
        compressed_upsampled_temporal=compressed_upsampled_temporal,
        template_shift_index=template_shift_index,
        upsampled_shifted_template_index=upsampled_shifted_template_index,
        geom=geom,
        conv_ignore_threshold=conv_ignore_threshold,
        coarse_approx_error_threshold=coarse_approx_error_threshold,
        max_shift="full",
        conv_batch_size=conv_batch_size,
        units_batch_size=units_batch_size,
        device=device,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )

    pconv_index = np.zeros(
        (
            template_shift_index.n_shifted_templates,
            upsampled_shifted_template_index.n_upsampled_shifted_templates,
        ),
        dtype=int,
    )
    n_pconvs = 1
    with h5py.File(output_hdf5_filename, "w") as h5:
        # resizeable pconv dataset
        spike_length_samples = template_data.templates.shape[1]
        pconv = h5.create_dataset(
            "pconv",
            dtype=np.float32,
            shape=(1, 2 * spike_length_samples - 1),
            maxshape=(None, 2 * spike_length_samples - 1),
            chunks=(128, 2 * spike_length_samples - 1),
        )

        for chunk_res in chunk_res_iterator:
            if chunk_res is None:
                continue

            # get shifted template indices for A
            shifted_temp_ix_a = template_shift_index.template_shift_index[
                chunk_res.template_indices_a,
                chunk_res.shift_indices_a,
            ]

            # upsampled shifted template indices for B
            up_shifted_temp_ix_b = (
                upsampled_shifted_template_index.upsampled_shifted_template_index[
                    chunk_res.template_indices_b,
                    chunk_res.shift_indices_b,
                    chunk_res.upsampling_indices_b,
                ]
            )

            # store new set of indices
            new_pconv_indices = chunk_res.compression_index + n_pconvs
            pconv_index[shifted_temp_ix_a, up_shifted_temp_ix_b] = new_pconv_indices

            # store new pconvs
            n_new_pconvs = chunk_res.compressed_conv.shape[0]
            pconv.resize(n_pconvs + n_new_pconvs, axis=0)
            pconv[n_pconvs:] = chunk_res.compressed_conv

            n_pconvs += n_new_pconvs

        # write fixed size outputs
        h5.create_dataset("shifts", data=template_shift_index.all_pitch_shifts)
        h5.create_dataset(
            "shifted_template_index", data=template_shift_index.template_shift_index
        )
        h5.create_dataset(
            "upsampled_shifted_template_index",
            data=upsampled_shifted_template_index.upsampled_shifted_template_index,
        )
        h5.create_dataset("pconv_index", data=pconv_index)

    return output_hdf5_filename


def iterate_compressed_pairwise_convolutions(
    template_data: templates.TemplateData,
    low_rank_templates: template_util.LowRankTemplates,
    compressed_upsampled_temporal: template_util.CompressedUpsampledTemplates,
    template_shift_index: drift_util.TemplateShiftIndex,
    upsampled_shifted_template_index: UpsampledShiftedTemplateIndex,
    geom: Optional[np.ndarray] = None,
    conv_ignore_threshold=0.0,
    coarse_approx_error_threshold=0.0,
    max_shift="full",
    conv_batch_size=1024,
    units_batch_size=8,
    device=None,
    n_jobs=0,
    show_progress=True,
) -> Iterator[Optional[CompressedConvResult]]:
    """A generator of CompressedConvResults capturing all pairs of templates


    Runs the function compressed_convolve_pairs on chunks of units.

    This is a helper function for parallelizing computation of cross correlations
    between pairs of templates. There are too many to store all the results in
    memory, so this is a generator yielding a chunk at a time. Callers may
    process the results differently.
    """
    # construct drift-related helper data if needed
    print(f"iterate_compressed_pairwise_convolutions {conv_batch_size=} {units_batch_size=} {device=}")
    n_shifts = template_shift_index.all_pitch_shifts.size
    do_shifting = n_shifts > 1
    geom_kdtree = reg_geom_kdtree = match_distance = None
    reg_geom = template_data.registered_geom
    if do_shifting:
        assert geom is not None
        assert reg_geom is not None
        geom_kdtree = KDTree(geom)
        reg_geom_kdtree = KDTree(reg_geom)
        match_distance = pdist(geom).min() / 2

    # make chunks
    units = np.unique(template_data.unit_ids)
    jobs = []
    for start_a in range(0, units.size, units_batch_size):
        end_a = min(start_a + units_batch_size, units.size)
        for start_b in range(start_a, units.size, units_batch_size):
            end_b = min(start_b + units_batch_size, units.size)
            jobs.append((units[start_a:end_a], units[start_b:end_b]))

    # worker kwargs
    kwargs = dict(
        template_data=template_data,
        low_rank_templates=low_rank_templates,
        compressed_upsampled_temporal=compressed_upsampled_temporal,
        template_shift_index=template_shift_index,
        upsampled_shifted_template_index=upsampled_shifted_template_index,
        geom=geom,
        reg_geom=reg_geom,
        geom_kdtree=geom_kdtree,
        reg_geom_kdtree=reg_geom_kdtree,
        match_distance=match_distance,
        conv_ignore_threshold=conv_ignore_threshold,
        coarse_approx_error_threshold=coarse_approx_error_threshold,
        max_shift=max_shift,
        batch_size=conv_batch_size,
    )

    n_jobs, Executor, context, rank_queue = get_pool(n_jobs, with_rank_queue=True)
    with Executor(
        n_jobs,
        mp_context=context,
        initializer=_conv_worker_init,
        initargs=(rank_queue, device, kwargs),
    ) as pool:
        it = pool.map(_conv_job, jobs)
        if show_progress:
            it = tqdm(
                it,
                smoothing=0.01,
                desc="Pairwise convolution",
                unit="pair block",
                total=len(jobs),
            )
        yield from it


@dataclass
class CompressedConvResult:
    """Return type of compressed_convolve_pairs

    After convolving a bunch of template pairs, some convolutions
    may be zero. Let n_pairs be the number of nonzero convolutions.
    We don't store the zero ones.
    """

    # arrays of shape n_pairs,
    # For each convolved pair, these document which templates were
    # in the pair, what their relative shifts were, and what the
    # upsampling was (we only upsample the RHS)
    template_indices_a: np.ndarray
    template_indices_b: np.ndarray
    shift_indices_a: np.ndarray
    shift_indices_b: np.ndarray
    upsampling_indices_b: np.ndarray

    # another one of shape n_pairs
    # maps a pair index to the corresponding convolution index
    # some convolutions are duplicates, so this array contains
    # many duplicate entries in the range 0, ..., n_convs-1
    compression_index: np.ndarray

    # this one has shape (n_convs, 2 * spike_length_samples - 1)
    compressed_conv: np.ndarray


def compressed_convolve_pairs(
    template_data: templates.TemplateData,
    low_rank_templates: template_util.LowRankTemplates,
    compressed_upsampled_temporal: template_util.CompressedUpsampledTemplates,
    template_shift_index: drift_util.TemplateShiftIndex,
    upsampled_shifted_template_index: UpsampledShiftedTemplateIndex,
    geom: Optional[np.ndarray] = None,
    reg_geom: Optional[np.ndarray] = None,
    geom_kdtree: Optional[KDTree] = None,
    reg_geom_kdtree: Optional[KDTree] = None,
    match_distance: Optional[float] = None,
    units_a: Optional[np.ndarray] = None,
    units_b: Optional[np.ndarray] = None,
    conv_ignore_threshold=0.0,
    coarse_approx_error_threshold=0.0,
    max_shift="full",
    batch_size=1024,
    device=None,
) -> Optional[CompressedConvResult]:
    """Compute compressed pairwise convolutions between template pairs

    Takes as input all the template data and groups of pairs of units to convolve
    (units_a,b). units_a,b are unit indices, not template indices (i.e., coarse
    units, not superresolved bin indices).

    Returns compressed convolutions between all units_a[i], units_b[j], for all
    shifts, superres templates, and upsamples. Some of these may be zero or may
    be duplicates, so the return value is a sparse representation. See below.
    """
    # print(f"compressed_convolve_pairs {device=}")
    # print(f"{units_a.shape=}")
    # print(f"{units_b.shape=}")
    # print(f"{(units_a.size * units_b.size)=}")
    # print(f"compressed_convolve_pairs {batch_size=} {units_a.size=} {device=}")

    # what pairs, shifts, etc are we convolving?
    shifted_temp_ix_a, temp_ix_a, shift_a, unit_a = handle_shift_indices(
        units_a, template_data.unit_ids, template_shift_index
    )
    shifted_temp_ix_b, temp_ix_b, shift_b, unit_b = handle_shift_indices(
        units_b, template_data.unit_ids, template_shift_index
    )
    # print(f"{shifted_temp_ix_a.shape=}")
    # print(f"{shifted_temp_ix_b.shape=}")

    # get (shifted) spatial components * singular values
    spatial_singular_a = get_shifted_spatial_singular(
        temp_ix_a,
        shift_a,
        template_shift_index,
        low_rank_templates,
        geom=geom,
        registered_geom=reg_geom,
        geom_kdtree=geom_kdtree,
        match_distance=match_distance,
        device=device,
    )
    spatial_singular_b = get_shifted_spatial_singular(
        temp_ix_b,
        shift_b,
        template_shift_index,
        low_rank_templates,
        geom=geom,
        registered_geom=reg_geom,
        geom_kdtree=geom_kdtree,
        match_distance=match_distance,
        device=device,
    )
    # print(f"{low_rank_templates.spatial_components.dtype=} {low_rank_templates.singular_values.dtype=}")
    # print(f"{compressed_upsampled_temporal.compressed_upsampled_templates.dtype=}")
    # print(f"{spatial_singular_a.dtype=} {spatial_singular_b.dtype=}")

    # figure out pairs of shifted templates to convolve in a deduplicated way
    pairs_ret = shift_deduplicated_pairs(
        shifted_temp_ix_a,
        shifted_temp_ix_b,
        spatial_singular_a,
        spatial_singular_b,
        temp_ix_a,
        temp_ix_b,
        shift_a=shift_a,
        shift_b=shift_b,
        template_shift_index=template_shift_index,
        conv_ignore_threshold=conv_ignore_threshold,
        geom=geom,
        registered_geom=reg_geom,
        reg_geom_kdtree=reg_geom_kdtree,
        match_distance=match_distance,
    )
    if pairs_ret is None:
        return None
    ix_a, ix_b, compression_index, conv_ix = pairs_ret
    # print(f"A {ix_a.shape=}")
    # print(f"A {ix_b.shape=}")
    # print(f"A {compression_index.shape=}")
    # print(f"A {conv_ix.shape=}")

    # print(f"-----------")
    # print(f"after pairs {conv_ix.shape=} {compression_index.shape=}")
    # print(f"{compression_index.min()=} {compression_index.max()=}")
    # print(f"{ix_a.shape=} {ix_b.shape=}")

    # handle upsampling
    # each pair will be duplicated by the b unit's number of upsampled copies
    (
        ix_a,
        ix_b,
        compression_index,
        conv_ix,
        conv_upsampling_indices_b,
        conv_temporal_components_up_b,
    ) = compressed_upsampled_pairs(
        ix_a,
        ix_b,
        compression_index,
        conv_ix,
        temp_ix_b,
        shifted_temp_ix_b,
        upsampled_shifted_template_index,
        compressed_upsampled_temporal,
    )
    # print(f"B {ix_a.shape=}")
    # print(f"B {ix_b.shape=}")
    # print(f"B {compression_index.shape=}")
    # print(f"B {conv_ix.shape=}")

    # print(f"-----------")
    # print(f"after up {conv_ix.shape=} {compression_index.shape=}")
    # print(f"{compression_index.min()=} {compression_index.max()=}")
    # print(f"{ix_a.shape=} {ix_b.shape=}")

    # # now, these arrays all have length n_pairs
    # shifted_temp_ix_a = shifted_temp_ix_a[ix_a]
    # temp_ix_a = temp_ix_a[ix_a]
    # shift_a = shift_a[ix_a]
    # shifted_temp_ix_b = shifted_temp_ix_b[ix_b]
    # temp_ix_b = temp_ix_b[ix_b]
    # shift_b = shift_b[ix_b]

    # run convolutions
    temporal_a = low_rank_templates.temporal_components[temp_ix_a]
    # print(f"{spatial_singular_a[ix_a[conv_ix]].shape=}")
    # print(f"{spatial_singular_b[ix_b[conv_ix]].shape=}")
    # print(f"{temporal_a[ix_a[conv_ix]].shape=}")
    # print(f"{conv_temporal_components_up_b.shape=}")
    pconv, kept = correlate_pairs_lowrank(
        torch.as_tensor(spatial_singular_a[ix_a[conv_ix]], device=device),
        torch.as_tensor(spatial_singular_b[ix_b[conv_ix]], device=device),
        torch.as_tensor(temporal_a[ix_a[conv_ix]], device=device),
        torch.as_tensor(conv_temporal_components_up_b, device=device),
        max_shift=max_shift,
        conv_ignore_threshold=conv_ignore_threshold,
        batch_size=batch_size,
    )
    # print(f"-----------")
    # print(f"after corr {pconv.shape=} {conv_ix[kept].shape=}")
    conv_ix = conv_ix[kept]
    if not conv_ix.size:
        return None
    kept_pairs = np.flatnonzero(np.isin(compression_index, kept))
    # print(f"-----------")
    # print(f"kept {pconv.shape=} {conv_ix.shape=} {compression_index.shape=}")
    # print(f"{compression_index.min()=} {compression_index.max()=}")
    # print(f"{compression_index[kept_pairs].min()=} {compression_index[kept_pairs].max()=}")
    # print(f"{ix_a.shape=} {ix_b.shape=}")
    # print(f"{kept.shape=} {kept.dtype=} {kept.min()=} {kept.max()=}")
    # print(f"{kept_pairs.shape=} {kept_pairs.dtype=} {kept_pairs.min()=} {kept_pairs.max()=}")
    compression_index = np.searchsorted(kept, compression_index[kept_pairs])
    conv_ix = np.searchsorted(kept_pairs, conv_ix)
    ix_a = ix_a[kept_pairs]
    ix_b = ix_b[kept_pairs]
    # compression_index = compression_index[kept]
    pconv = pconv.cpu()
    # print(f"-----------")
    # print(f"after searchsorted {pconv.shape=} {conv_ix.shape=} {compression_index.shape=}")
    # print(f"{compression_index.min()=} {compression_index.max()=}")
    # print(f"{ix_a.shape=} {ix_b.shape=}")

    # coarse approx
    # print(f"-----------")
    # print(f"before approx {pconv.shape=} {conv_ix.shape=} {compression_index.shape=}")
    pconv, old_ix_to_new_ix = coarse_approximate(
        pconv,
        unit_a[ix_a[conv_ix]],
        unit_b[ix_b[conv_ix]],
        temp_ix_a[ix_a[conv_ix]],
        shift_a[ix_a[conv_ix]],
        shift_b[ix_b[conv_ix]],
        coarse_approx_error_threshold=coarse_approx_error_threshold,
    )
    # print(f"-----------")
    # print(f"after approx")
    # print(f"{pconv.shape=} {conv_ix.shape=} {old_ix_to_new_ix.shape=} {compression_index.shape=}")
    # print(f"{compression_index.min()=} {compression_index.max()=}")
    # print(f"{old_ix_to_new_ix.min()=} {old_ix_to_new_ix.max()=}")
    compression_index = old_ix_to_new_ix[compression_index]
    # above function invalidates the whole idea of conv_ix
    del conv_ix

    # recover metadata
    temp_ix_a = temp_ix_a[ix_a]
    shift_ix_a = np.searchsorted(template_shift_index.all_pitch_shifts, shift_a[ix_a])
    temp_ix_b = temp_ix_b[ix_b]
    shift_ix_b = np.searchsorted(template_shift_index.all_pitch_shifts, shift_b[ix_b])

    return CompressedConvResult(
        template_indices_a=temp_ix_a,
        template_indices_b=temp_ix_b,
        shift_indices_a=shift_ix_a,
        shift_indices_b=shift_ix_b,
        upsampling_indices_b=conv_upsampling_indices_b[compression_index],
        compression_index=compression_index,
        compressed_conv=pconv.numpy(),
    )


# -- helpers


def correlate_pairs_lowrank(
    spatial_a,
    spatial_b,
    temporal_a,
    temporal_b,
    max_shift="full",
    conv_ignore_threshold=0.0,
    batch_size=1024,
):
    """Convolve pairs of low rank templates

    For each i, we want to convolve (temporal_a[i] @ spatial_a[i]) with
    (temporal_b[i] @ spatial_b[i]). So, spatial_{a,b} and temporal_{a,b}
    should contain lots of duplicates, since they are already representing
    pairs.

    Templates Ka = Sa Ta, Kb = Sb Tb. The channel-summed convolution is
        (Ka (*) Kb) = sum_c Ka(c) * Kb(c)
                    = (Sb.T @ Ka) (*) Tb
                    = (Sb.T @ Sa @ Ta) (*) Tb
    where * is cross-correlation, and (*) is channel (or rank) summed.
    We use full-height conv2d to do rank-summed convs.

    Returns
    -------
    pconv, kept
    """
    n_pairs, rank, nchan = spatial_a.shape
    n_pairs_, rank_, nchan_ = spatial_b.shape
    assert rank == rank_
    assert nchan == nchan_
    assert n_pairs == n_pairs_
    n_pairs_, t, rank_ = temporal_a.shape
    assert n_pairs == n_pairs_
    assert rank_ == rank
    n_pairs_, t_, rank_ = temporal_b.shape
    assert n_pairs == n_pairs_
    assert t == t_
    assert rank == rank_
    # print(f"{spatial_a.device=} {spatial_b.device=} {temporal_a.device=} {temporal_b.device=}")
    # print(f"compressed_convolve_pairs {batch_size=} {n_pairs=} {spatial_a.device=}")

    if max_shift == "full":
        max_shift = t - 1
    elif max_shift == "valid":
        max_shift = 0
    elif max_shift == "same":
        max_shift = t // 2

    # batch over n_pairs for memory reasons
    pconv = torch.zeros(
        (n_pairs, 2 * max_shift + 1), dtype=spatial_a.dtype, device=spatial_a.device
    )
    for istart in range(0, n_pairs, batch_size):
        iend = min(istart + batch_size, n_pairs)
        ix = slice(istart, iend)

        # want conv filter: nco, 1, rank, t
        template_a = torch.bmm(temporal_a[ix], spatial_a[ix])
        conv_filt = torch.bmm(spatial_b[ix], template_a.mT)
        conv_filt = conv_filt[:, None]  # (nco, 1, rank, t)

        # 1, nco, rank, t
        conv_in = temporal_b[ix].mT[None]

        # conv2d:
        # depthwise, chans=nco. batch=1. h=rank. w=t. out: nup=1, nco, 1, 2p+1.
        # input (conv_in): nup, nco, rank, t.
        # filters (conv_filt): nco, 1, rank, t. (groups=nco).
        pconv_ = F.conv2d(
            conv_in, conv_filt, padding=(0, max_shift), groups=iend - istart
        )
        pconv[istart:iend] = pconv_[0, :, 0, :]  # nco, nup, time

    # more stringent covisibility
    if conv_ignore_threshold > 0:
        max_val = pconv.reshape(n_pairs, -1).abs().max(dim=1).values
        kept = max_val > conv_ignore_threshold
        pconv = pconv[kept]
        kept = np.flatnonzero(kept.numpy(force=True))
    else:
        kept = np.arange(len(pconv))

    return pconv, kept


def handle_shift_indices(units, unit_ids, template_shift_index):
    shifted_temp_ix_to_unit = unit_ids[template_shift_index.shifted_temp_ix_to_temp_ix]
    if units is None:
        shifted_temp_ix = np.arange(template_shift_index.n_shifted_templates)
    else:
        shifted_temp_ix = np.flatnonzero(np.isin(shifted_temp_ix_to_unit, units))

    shift = template_shift_index.shifted_temp_ix_to_shift[shifted_temp_ix]
    temp_ix = template_shift_index.shifted_temp_ix_to_temp_ix[shifted_temp_ix]
    unit = unit_ids[temp_ix]

    return shifted_temp_ix, temp_ix, shift, unit


def get_shifted_spatial_singular(
    temp_ix,
    shift,
    template_shift_index,
    low_rank_templates,
    geom=None,
    registered_geom=None,
    geom_kdtree=None,
    match_distance=None,
    device=None,
):
    # do we need to shift the templates?
    n_shifts = template_shift_index.all_pitch_shifts.size
    do_shifting = n_shifts > 1

    spatial_singular = (
        low_rank_templates.spatial_components[temp_ix]
        * low_rank_templates.singular_values[temp_ix][..., None]
    )
    if do_shifting:
        spatial_singular = drift_util.get_waveforms_on_static_channels(
            spatial_singular,
            registered_geom,
            n_pitches_shift=shift,
            registered_geom=geom,
            target_kdtree=geom_kdtree,
            match_distance=match_distance,
            fill_value=0.0,
        )
    spatial_singular = torch.as_tensor(spatial_singular, device=device)

    return spatial_singular


def shift_deduplicated_pairs(
    shifted_temp_ix_a,
    shifted_temp_ix_b,
    spatialsing_a,
    spatialsing_b,
    temp_ix_a,
    temp_ix_b,
    shift_a=None,
    shift_b=None,
    template_shift_index=None,
    conv_ignore_threshold=0.0,
    geom=None,
    registered_geom=None,
    reg_geom_kdtree=None,
    match_distance=None,
):
    """Choose a set of pairs of indices from group A and B to convolve

    Some pairs of shifted templates don't overlap, so we don't need to convolve them.
    Some pairs of shifted templates never show up in the recording at the same time
    (what this code calls "cooccurrence"), so we don't need to convolve them.

    More complicated: for each shift, a certain set of registered template channels
    survives. Given that the some set of visible channels has survived for a pair of
    templates at shifts shift_a and shift_b, their cross-correlation at these shifts
    is the same as the one at shift_a_prime and shift_b_prime if the same exact channels
    survived at shift_a_prime and shift_b_prime and if
        shift_a-shift_b == shift_a_prime-shift_b_prime.

    Returns
    -------
    pair_ix_a, pair_ix_b
        Size < original number of shifted templates a,b
        The indices of shifted templates which overlap enough to be
        co-visible. So, these are subsets of shifted_temp_ix_a,b
    compression_index
        Size == pair_ix_a,b size
        Arrays with shape matching pair_ix_a,b, so that the xcorr of templates
        shifted_temp_ix_a[pair_ix_a[i]], shifted_temp_ix_b[pair_ix_b[i]]
        is the same as that of
        shifted_temp_ix_a[pair_ix_a[conv_ix[compression_index[i]]],
                          pair_ix_b[conv_ix[compression_index[i]]]
    conv_ix
        Size < original number of shifted templates a,b
        Pairs of templates which should actually be convolved
    """
    # check spatially overlapping
    chan_amp_a = torch.sqrt(torch.square(spatialsing_a).sum(1))
    chan_amp_b = torch.sqrt(torch.square(spatialsing_b).sum(1))
    pair = chan_amp_a @ chan_amp_b.T
    pair = pair > conv_ignore_threshold
    pair = pair.cpu()
    # print(f"___ after overlaps {pair.sum()=}")

    # co-occurrence
    cooccurrence = template_shift_index.cooccurrence[
        shifted_temp_ix_a[:, None],
        shifted_temp_ix_b[None, :],
    ]
    pair *= torch.as_tensor(cooccurrence, device=pair.device)
    # print(f"___ after cooccur {pair.sum()=}")

    pair_ix_a, pair_ix_b = torch.nonzero(pair, as_tuple=True)
    nco = pair_ix_a.numel()
    if not nco:
        return None
    # print(f"___ {nco=}")

    # if no shifting, deduplication is the identity
    do_shifting = template_shift_index.all_pitch_shifts.size > 1
    if not do_shifting:
        nco_range = torch.arange(nco, device=pair_ix_a.device)
        return pair_ix_a, pair_ix_b, nco_range, nco_range

    # shift deduplication. algorithm:
    # 1 for each shifted template, determine the set of registered channels
    #   which it occupies
    # 2 assign each such set an ID (an "active channel ID")
    #   - // then a pair of shifted templates' xcorr is a function of the pair
    #     // of active channel IDs and the difference of shifts
    # 3 figure out the set of unique (active chan id a, active chan id b, shift diff a,b)
    #   combinations in each pair of units

    # 1: get active channel neighborhoods as many-hot len(reg_geom)-vectors
    active_chans_a = drift_util.get_waveforms_on_static_channels(
        (chan_amp_a > 0).numpy(force=True),
        geom,
        n_pitches_shift=-shift_a,
        registered_geom=registered_geom,
        target_kdtree=reg_geom_kdtree,
        match_distance=match_distance,
        fill_value=0,
    )
    active_chans_b = drift_util.get_waveforms_on_static_channels(
        (chan_amp_b > 0).numpy(force=True),
        geom,
        n_pitches_shift=-shift_b,
        registered_geom=registered_geom,
        target_kdtree=reg_geom_kdtree,
        match_distance=match_distance,
        fill_value=0,
    )
    # 2: assign IDs to each such vector
    chanset_a, active_chan_ids_a = np.unique(
        active_chans_a, axis=0, return_inverse=True
    )
    chanset_b, active_chan_ids_b = np.unique(
        active_chans_b, axis=0, return_inverse=True
    )
    # print(f"___ {chanset_a.sum(1)=}")
    # print(f"___ {chanset_b.sum(1)=}")
    # print(f"___ {active_chan_ids_a.shape=} {np.unique(active_chan_ids_a).shape=}")
    # print(f"___ {active_chan_ids_b.shape=} {np.unique(active_chan_ids_b).shape=}")

    # 3
    temp_ix_a = temp_ix_a[pair_ix_a]
    temp_ix_b = temp_ix_b[pair_ix_b]
    # get the relative shifts
    shift_a = shift_a[pair_ix_a]
    shift_b = shift_b[pair_ix_b]
    shift_diff = shift_a - shift_b
    # print(f"{temp_ix_a=}")
    # print(f"{shift_a=}")
    # print(f"{active_chan_ids_a[pair_ix_a]=}")
    # print(f"{temp_ix_b=}")
    # print(f"{shift_b=}")
    # print(f"{active_chan_ids_b[pair_ix_b]=}")
    # print(f"{shift_diff=}")

    # figure out combinations
    conv_determiners = np.c_[
        temp_ix_a,
        active_chan_ids_a[pair_ix_a],
        temp_ix_b,
        active_chan_ids_b[pair_ix_b],
        shift_diff,
    ]
    # print(f"{conv_determiners=}")
    # conv_ix: indices of unique determiners
    # compression_index: which representative does each pair belong to
    _, conv_ix, compression_index = np.unique(
        conv_determiners, axis=0, return_index=True, return_inverse=True
    )

    return pair_ix_a, pair_ix_b, compression_index, conv_ix


UpsampledShiftedTemplateIndex = namedtuple(
    "UpsampledShiftedTemplateIndex",
    [
        "n_upsampled_shifted_templates",
        "upsampled_shifted_template_index",
        "up_shift_temp_ix_to_shift_temp_ix",
        "up_shift_temp_ix_to_temp_ix",
        "up_shift_temp_ix_to_comp_up_ix",
    ],
)


def get_upsampled_shifted_template_index(
    template_shift_index, compressed_upsampled_temporal
):
    """Make a compressed index space for upsampled shifted templates

    See also: template_util.{compressed_upsampled_templates,ComptessedUpsampledTemplates}.

    The comp_up_ix / compressed upsampled template indices here are indices into that
    structure.

    Returns
    -------
    UpsampledShiftedTemplateIndex
        named tuple with fields:
        upsampled_shifted_template_index : (n_templates, n_shifts, up_factor)
            Maps template_ix, shift_ix, up_ix -> compressed upsampled template index
        up_shift_temp_ix_to_shift_temp_ix
        up_shift_temp_ix_to_temp_ix
        up_shift_temp_ix_to_comp_up_ix
    """
    n_shifted_templates = template_shift_index.n_shifted_templates
    n_templates, n_shifts = template_shift_index.template_shift_index.shape
    max_upsample = compressed_upsampled_temporal.compressed_upsampling_map.shape[1]

    cur_up_shift_temp_ix = 0
    # fill with an invalid index
    upsampled_shifted_template_index = np.full(
        (n_templates, n_shifts, max_upsample), n_shifted_templates * max_upsample
    )
    usti2sti = []
    usti2ti = []
    usti2cui = []
    for i in range(n_templates):
        shifted_temps = template_shift_index.template_shift_index[i]
        valid_shifts = np.flatnonzero(shifted_temps < n_shifted_templates)

        upsampled_temps = compressed_upsampled_temporal.compressed_upsampling_map[i]
        unique_comp_up_inds, inverse = np.unique(upsampled_temps, return_inverse=True)

        for j in valid_shifts:
            up_shift_inds = cur_up_shift_temp_ix + np.arange(unique_comp_up_inds.size)
            upsampled_shifted_template_index[i, j] = up_shift_inds[inverse]
            cur_up_shift_temp_ix += up_shift_inds.size

            usti2sti.extend([shifted_temps[j]] * up_shift_inds.size)
            usti2ti.extend([i] * up_shift_inds.size)
            usti2cui.extend(unique_comp_up_inds)

    up_shift_temp_ix_to_shift_temp_ix = np.array(usti2sti)
    up_shift_temp_ix_to_temp_ix = np.array(usti2ti)
    up_shift_temp_ix_to_comp_up_ix = np.array(usti2cui)

    return UpsampledShiftedTemplateIndex(
        up_shift_temp_ix_to_shift_temp_ix.size,
        upsampled_shifted_template_index,
        up_shift_temp_ix_to_shift_temp_ix,
        up_shift_temp_ix_to_temp_ix,
        up_shift_temp_ix_to_comp_up_ix,
    )


def compressed_upsampled_pairs(
    ix_a,
    ix_b,
    compression_index,
    conv_ix,
    temp_ix_b,
    shifted_temp_ix_b,
    upsampled_shifted_template_index,
    compressed_upsampled_temporal,
):
    """Add in upsampling to the set of pairs that need to be convolved

    So far, ix_a,b, compression_index, and conv_ix are such that non-upsampled
    convolutions between templates ix_a[i], ix_b[i] equal that between templates
    ix_a[conv_ix[compression_index[i]]], ix_b[conv_ix[compression_index[i]]].

    We will upsample the templates in the RHS (b) in a compressed way.
    """
    up_factor = compressed_upsampled_temporal.compressed_upsampling_map.shape[1]
    if up_factor == 1:
        upinds = np.zeros(len(conv_ix), dtype=int)
        temp_comps = compressed_upsampled_temporal.compressed_upsampled_templates[
            temp_ix_b[ix_b[conv_ix]]
        ]
        return ix_a, ix_b, compression_index, conv_ix, upinds, temp_comps

    # each conv_ix needs to be duplicated as many times as its b template has
    # upsampled copies. And, all ix_{a,b}[i] such that compression_ix[i] lands in
    # that conv_ix need to be duplicated as well.
    ix_a_up = []
    ix_b_up = []
    compression_index_up = []
    conv_ix_up = []
    conv_compressed_upsampled_ix = []
    cur_dedup_ix = 0
    for i, convi in enumerate(conv_ix):
        # get b's shifted template ix
        conv_shifted_temp_ix_b = shifted_temp_ix_b[ix_b[convi]]

        # which compressed upsampled indices match this?
        which_up = np.flatnonzero(
            upsampled_shifted_template_index.up_shift_temp_ix_to_shift_temp_ix
            == conv_shifted_temp_ix_b
        )
        conv_comp_up_ix = (
            upsampled_shifted_template_index.up_shift_temp_ix_to_comp_up_ix[which_up]
        )

        # which deduplication indices map ix_a,b to this convi?
        which_dedup = np.flatnonzero(compression_index == i)

        # extend arrays with new indices
        nupi = conv_comp_up_ix.size
        ix_a_up.extend(np.repeat(ix_a[which_dedup], nupi))
        ix_b_up.extend(np.repeat(ix_b[which_dedup], nupi))
        conv_ix_up.extend([convi] * nupi)
        compression_index_up.extend(
            np.tile(np.arange(cur_dedup_ix, cur_dedup_ix + nupi), which_dedup.size)
        )
        cur_dedup_ix += nupi
        conv_compressed_upsampled_ix.extend(conv_comp_up_ix)

    ix_a_up = np.array(ix_a_up)
    ix_b_up = np.array(ix_b_up)
    compression_index_up = np.array(compression_index_up)
    conv_ix_up = np.array(conv_ix_up)
    conv_compressed_upsampled_ix = np.array(conv_compressed_upsampled_ix)

    # which upsamples and which templates?
    conv_upsampling_indices_b = (
        compressed_upsampled_temporal.compressed_index_to_upsampling_index[
            conv_compressed_upsampled_ix
        ]
    )
    conv_temporal_components_up_b = (
        compressed_upsampled_temporal.compressed_upsampled_templates[
            conv_compressed_upsampled_ix
        ]
    )

    return (
        ix_a_up,
        ix_b_up,
        compression_index_up,
        conv_ix_up,
        conv_upsampling_indices_b,
        conv_temporal_components_up_b,
    )


def coarse_approximate(
    pconv,
    units_a,
    units_b,
    temp_ix_a,
    shift_a,
    shift_b,
    coarse_approx_error_threshold=0.0,
):
    """Try to replace fine (superres+temporally upsampled) convs with coarse ones

    For each pair of convolved units, we first try to replace all of the pairwise
    convolutions between these units with their mean, respecting the shifts.

    If that fails, we try to do this in a factorized way: for each superres unit a,
    try to replace all of its convolutions with unit b with their mean, respecting
    the shifts.

    Above, "respecting the shifts" means we only do this within each shift-deduplication
    class, since changes in the sets of channels being convolved cause large changes
    in the cross correlation. pconv has already been deduplicated with respect to
    equivalent channel neighborhoods, so all that matters for that purpose is the
    shift difference.

    This needs to tell the caller how to update its bookkeeping.
    """
    new_pconv = []
    old_ix_to_new_ix = np.full(len(pconv), -1)
    cur_new_ix = 0
    shift_diff = shift_a - shift_b
    for ua in np.unique(units_a):
        ina = np.flatnonzero(units_a == ua)
        partners_b = np.unique(units_b[ina])
        for ub in partners_b:
            inab = ina[units_b[ina] == ub]
            dshift = shift_diff[inab]
            for shift in np.unique(dshift):
                inshift = inab[dshift == shift]

                convs = pconv[inshift]
                meanconv = convs.mean(dim=0, keepdims=True)
                if (convs - meanconv).abs().max() < coarse_approx_error_threshold:
                    # do something
                    new_pconv.append(meanconv)
                    old_ix_to_new_ix[inshift] = cur_new_ix
                    cur_new_ix += 1
                    continue
                # else:
                #     # if we don't want the factorized thing...
                #     new_pconv.append(convs)
                #     old_ix_to_new_ix[inshift] = np.arange(cur_new_ix, cur_new_ix + inshift.size)
                #     cur_new_ix += inshift.size
                #     continue

                active_temp_a = temp_ix_a[inshift]
                unique_active_temp_a = np.unique(active_temp_a)
                if unique_active_temp_a.size == 1:
                    new_pconv.append(convs)
                    old_ix_to_new_ix[inshift] = np.arange(
                        cur_new_ix, cur_new_ix + inshift.size
                    )
                    cur_new_ix += inshift.size
                    continue

                for tixa in unique_active_temp_a:
                    insup = active_temp_a == tixa
                    supconvs = convs[insup]

                    meanconv = supconvs.mean(dim=0, keepdims=True)
                    if (convs - meanconv).abs().max() < coarse_approx_error_threshold:
                        new_pconv.append(meanconv)
                        old_ix_to_new_ix[inshift[insup]] = cur_new_ix
                        cur_new_ix += 1
                    else:
                        new_pconv.append(supconvs)
                        old_ix_to_new_ix[inshift[insup]] = np.arange(
                            cur_new_ix, cur_new_ix + insup.sum()
                        )
                        cur_new_ix += insup.sum()

    new_pconv = torch.cat(new_pconv)
    return new_pconv, old_ix_to_new_ix


# -- parallelism helpers


@dataclass
class ConvWorkerContext:
    template_data: templates.TemplateData
    low_rank_templates: template_util.LowRankTemplates
    compressed_upsampled_temporal: template_util.CompressedUpsampledTemplates
    template_shift_index: drift_util.TemplateShiftIndex
    upsampled_shifted_template_index: UpsampledShiftedTemplateIndex
    geom: Optional[np.ndarray] = None
    reg_geom: Optional[np.ndarray] = None
    geom_kdtree: Optional[KDTree] = None
    reg_geom_kdtree: Optional[KDTree] = None
    match_distance: Optional[float] = None
    conv_ignore_threshold: float = 0.0
    coarse_approx_error_threshold: float = 0.0
    max_shift: Union[int, str] = "full"
    batch_size: int = 128
    device: Optional[torch.device] = None


_conv_worker_context = None


def _conv_worker_init(rank_queue, device, kwargs):
    global _conv_worker_context

    my_rank = rank_queue.get()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda", index=my_rank % torch.cuda.device_count())

    _conv_worker_context = ConvWorkerContext(device=device, **kwargs)


def _conv_job(unit_chunk):
    global _conv_worker_context
    units_a, units_b = unit_chunk
    return compressed_convolve_pairs(
        units_a=units_a, units_b=units_b, **asdict_shallow(_conv_worker_context)
    )


def asdict_shallow(obj):
    return {field.name: getattr(obj, field.name) for field in fields(obj)}
