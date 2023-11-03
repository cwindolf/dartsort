from dataclasses import dataclass, fields
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from dartsort.templates import template_util
from dartsort.util import drift_util
from dartsort.util.multiprocessing_util import get_pool
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from tqdm.auto import tqdm

# todo: extend this code to also handle computing pairwise
#       stuff necessary for the merge! ie shifts, scaling,
#       residnorm(a,b) (or min of rn(a,b),rn(b,a)???)


def sparse_pairwise_conv(
    output_hdf5_filename,
    geom,
    template_data,
    template_temporal_components,
    template_upsampled_temporal_components,
    template_singular_values,
    template_spatial_components,
    chunk_time_centers_s=None,
    motion_est=None,
    conv_ignore_threshold: float = 0.0,
    coarse_approx_error_threshold: float = 0.0,
    min_channel_amplitude: float = 1.0,
    units_per_chunk=8,
    overwrite=False,
    show_progress=True,
    device=None,
    n_jobs=0,
):
    """

    Arguments
    ---------
    template_* : tensors or arrays
        template SVD approximations
    conv_ignore_threshold: float = 0.0
        pairs will be ignored (i.e., pconv set to 0) if their pconv
        does not exceed this value
    coarse_approx_error_threshold: float = 0.0
        superres will not be used if coarse pconv and superres pconv
        are uniformly closer than this threshold value

    Returns
    -------
    pitch_shifts : array
        array of all the pitch shifts
        use searchsorted to find the pitch shift ix for a pitch shift
    index_table: torch sparse tensor
        index_table[(pitch shift ix a, superres label a, pitch shift ix b, superres label b)] = (
            0
            if superres pconv a,b at these shifts was below the conv_ignore_threshold
            else pconv_index)
    pconvs: np.ndarray
        pconv[pconv_index] is a cross-correlation of two templates, summed over chans
    """
    if overwrite:
        pass

    (
        n_templates,
        spike_length_samples,
        upsampling_factor,
    ) = template_upsampled_temporal_components.shape[:3]

    # find all of the co-occurring pitch shift and template pairs
    temp_shift_index = get_shift_and_unit_pairs(
        chunk_time_centers_s,
        geom,
        template_data,
        motion_est=motion_est,
    )

    # check if the convolutions need to be drift-aware
    # they do if we need to do any channel selection
    is_drifting = not np.array_equal(temp_shift_index.all_pitch_shifts, [0])
    if template_data.registered_geom is not None:
        is_drifting |= not np.array_equal(geom, template_data.registered_geom)

    # initialize pairwise conv data structures
    # index_table[shifted_temp_ix(i), shifted_temp_ix(j)] = pconvix(i,j)
    pair_index_table = np.zeros(
        (temp_shift_index.n_shifted_templates, temp_shift_index.n_shifted_templates),
        dtype=int,
    )
    upsampling_index_table = np.zeros(
        (temp_shift_index.n_shifted_templates, temp_shift_index.n_shifted_templates),
        dtype=int,
    )
    # pconvs[pconvix(i,j)] = (2*spikelen-1, upsampling_factor) arr of pconv(shifted_temp(i), shifted_temp(j))

    cur_pair_ix = 1
    cur_pconv_ix = 1
    with h5py.File(output_hdf5_filename, "w") as h5:
        # resizeable pconv dataset
        pconv = h5.create_dataset(
            "pconv",
            dtype=np.float32,
            shape=(1, upsampling_factor, 2 * spike_length_samples - 1),
            maxshape=(None, upsampling_factor, 2 * spike_length_samples - 1),
            chunks=(128, upsampling_factor, 2 * spike_length_samples - 1),
        )

        # pconv[0] is special -- it is 0.
        pconv[0] = 0.0

        # res is a ConvBatchResult
        for res in compute_pairwise_convs(
            template_data,
            template_spatial_components,
            template_singular_values,
            template_temporal_components,
            template_upsampled_temporal_components,
            temp_shift_index.shifted_temp_ix_to_temp_ix,
            temp_shift_index.shifted_temp_ix_to_shift,
            geom,
            cooccurrence=temp_shift_index.cooccurrence,
            conv_ignore_threshold=conv_ignore_threshold,
            coarse_approx_error_threshold=coarse_approx_error_threshold,
            min_channel_amplitude=min_channel_amplitude,
            is_drifting=is_drifting,
            units_per_chunk=units_per_chunk,
            n_jobs=n_jobs,
            device=device,
            show_progress=show_progress,
            max_shift="full",
            store_conv=True,
            compute_max=False,
        ):
            if res is None:
                continue
            new_pair_ix = res.pair_ix + cur_pair_ix
            pair_index_table[res.shifted_temp_ix_a, res.shifted_temp_ix_b] = new_pair_ix

            new_pconv_ix = res.pconv_ix + cur_pconv_ix
            upsampling_index_table[new_pair_ix, res.upsampling_ix] = new_pconv_ix

            pconv.resize(cur_pconv_ix + res.cconv_up.shape[0], axis=0)
            pconv[cur_pconv_ix:] = res.cconv_up
            cur_pconv_ix += res.cconv_up.shape[0]

        # smaller datasets all at once
        h5.create_dataset(
            "template_shift_index", data=temp_shift_index.template_shift_index
        )
        h5.create_dataset("pconv_index_table", data=pconv_index_table)
        h5.create_dataset("shifts", data=temp_shift_index.all_pitch_shifts)
        h5.create_dataset(
            "shifted_temp_ix_to_temp_ix",
            data=temp_shift_index.shifted_temp_ix_to_temp_ix,
        )
        h5.create_dataset(
            "shifted_temp_ix_to_shift", data=temp_shift_index.shifted_temp_ix_to_shift
        )
        h5.create_dataset(
            "shifted_temp_ix_to_unit",
            data=template_data.unit_ids[temp_shift_index.shifted_temp_ix_to_temp_ix],
        )

    return output_hdf5_filename


# -- main general worker function


def compute_pairwise_convs(
    template_data,
    spatial,
    singular,
    temporal,
    temporal_up,
    shifted_temp_ix_to_temp_ix,
    shifted_temp_ix_to_shift,
    geom,
    cooccurrence,
    conv_ignore_threshold=0.0,
    coarse_approx_error_threshold=0.0,
    min_channel_amplitude=1.0,
    max_shift="full",
    is_drifting=True,
    store_conv=True,
    compute_max=False,
    units_per_chunk=8,
    n_jobs=0,
    device=None,
    show_progress=True,
):
    # chunk up coarse unit ids, go by pairs of chunks
    units = np.unique(template_data.unit_ids)
    jobs = []
    for start_a in range(0, units.size, units_per_chunk):
        end_a = min(start_a + units_per_chunk, units.size)
        for start_b in range(start_a, units.size, units_per_chunk):
            end_b = min(start_b + units_per_chunk, units.size)
            jobs.append((units[start_a:end_a], units[start_b:end_b]))
    if show_progress:
        jobs = tqdm(
            jobs, smoothing=0.01, desc="Pairwise convolution", unit="pair block"
        )

    # compute the coarse templates if needed
    if units.size == template_data.unit_ids.size:
        # coarse templates are original templates
        coarse_approx_error_threshold = 0
    if coarse_approx_error_threshold > 0:
        coarse_templates = template_util.weighted_average(
            template_data.unit_ids, template_data.templates, template_data.spike_counts
        )
        (
            coarse_temporal,
            coarse_singular,
            coarse_spatial,
        ) = template_util.svd_compress_templates(
            coarse_templates,
            rank=singular.shape[1],
            min_channel_amplitude=min_channel_amplitude,
        )

    # template data to torch
    spatial_singular = torch.as_tensor(spatial * singular[:, :, None])
    temporal = torch.as_tensor(temporal)
    temporal_up = torch.as_tensor(temporal_up)
    if coarse_approx_error_threshold > 0:
        coarse_spatial_singular = torch.as_tensor(
            coarse_spatial * coarse_singular[:, :, None]
        )
        coarse_temporal = torch.as_tensor(coarse_temporal)
    else:
        coarse_spatial_singular = None
        coarse_temporal = None

    n_jobs, Executor, context, rank_queue = get_pool(n_jobs, with_rank_queue=True)

    pconv_params = dict(
        store_conv=store_conv,
        compute_max=compute_max,
        is_drifting=is_drifting,
        max_shift=max_shift,
        conv_ignore_threshold=conv_ignore_threshold,
        coarse_approx_error_threshold=coarse_approx_error_threshold,
        spatial_singular=spatial_singular,
        temporal=temporal,
        temporal_up=temporal_up,
        coarse_spatial_singular=coarse_spatial_singular,
        coarse_temporal=coarse_temporal,
        unit_ids=template_data.unit_ids,
        shifted_temp_ix_to_shift=shifted_temp_ix_to_shift,
        shifted_temp_ix_to_temp_ix=shifted_temp_ix_to_temp_ix,
        shifted_temp_ix_to_unit=template_data.unit_ids[shifted_temp_ix_to_temp_ix],
        cooccurrence=cooccurrence,
        geom=geom,
        registered_geom=template_data.registered_geom,
    )

    with Executor(
        n_jobs,
        mp_context=context,
        initializer=_pairwise_conv_init,
        initargs=(device, rank_queue, pconv_params),
    ) as pool:
        yield from pool.map(_pairwise_conv_job, jobs)


# -- parallel job code


# helper class which stores parameters for _pairwise_conv_job
@dataclass
class PairwiseConvContext:
    device: torch.device

    # parameters
    store_conv: bool
    compute_max: bool
    is_drifting: bool
    max_shift: int
    conv_ignore_threshold: float
    coarse_approx_error_threshold: float

    # superres registered templates
    spatial_singular: torch.Tensor
    temporal: torch.Tensor
    temporal_up: torch.Tensor
    coarse_spatial_singular: Optional[torch.Tensor]
    coarse_temporal: Optional[torch.Tensor]
    cooccurrence: torch.Tensor

    # template indexing helper arrays
    unit_ids: np.ndarray
    shifted_temp_ix_to_temp_ix: np.ndarray
    shifted_temp_ix_to_shift: np.ndarray
    shifted_temp_ix_to_unit: np.ndarray

    # only needed if is_drifting
    geom: np.ndarray
    registered_geom: np.ndarray
    geom_kdtree: Optional[KDTree]
    reg_geom_kdtree: Optional[KDTree]
    match_distance: Optional[float]


_pairwise_conv_context = None


def _pairwise_conv_init(
    device,
    rank_queue,
    kwargs,
):
    global _pairwise_conv_context

    # figure out what device to work on
    my_rank = rank_queue.get()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda", index=my_rank % torch.cuda.device_count())

    # handle string max_shift
    max_shift = kwargs.pop("max_shift", "full")
    t = kwargs["temporal"].shape[1]
    if max_shift == "full":
        max_shift = t - 1
    elif max_shift == "valid":
        max_shift = 0
    elif max_shift == "same":
        max_shift = t // 2
    kwargs["max_shift"] = max_shift

    kwargs["geom_kdtree"] = kwargs["reg_geom_kdtree"] = kwargs["match_distance"] = None
    if kwargs["is_drifting"]:
        kwargs["geom_kdtree"] = KDTree(kwargs["geom"])
        kwargs["reg_geom_kdtree"] = KDTree(kwargs["registered_geom"])
        kwargs["match_distance"] = pdist(kwargs["geom"]).min() / 2

    _pairwise_conv_context = PairwiseConvContext(device=device, **kwargs)


@dataclass
class ConvBatchResult:
    # arrays of length <n convolved pairs>
    shifted_temp_ix_a: np.ndarray
    shifted_temp_ix_b: np.ndarray
    # array of length <n convolved pairs> such that the ith
    # pair's array of upsampled convs is cconv_up[cconv_ix[i]]
    cconv_ix: np.ndarray
    cconv_up: Optional[np.ndarray]
    max_conv: Optional[float]
    best_shift: Optional[int]


def _pairwise_conv_job(unit_chunk):
    global _pairwise_conv_context
    p = _pairwise_conv_context

    units_a, units_b = unit_chunk

    # this job consists of pairs of coarse units
    # lets get all shifted superres template indices corresponding to those pairs,
    # and the template indices, pitch shifts, and coarse units while we're at it
    shifted_temp_ix_a = np.flatnonzero(np.isin(p.shifted_temp_ix_to_unit, units_a))
    shifted_temp_ix_b = np.flatnonzero(np.isin(p.shifted_temp_ix_to_unit, units_b))
    temp_ix_a = p.shifted_temp_ix_to_temp_ix[shifted_temp_ix_a]
    temp_ix_b = p.shifted_temp_ix_to_temp_ix[shifted_temp_ix_b]
    shift_a = p.shifted_temp_ix_to_shift[shifted_temp_ix_a]
    shift_b = p.shifted_temp_ix_to_shift[shifted_temp_ix_b]
    unit_a = p.unit_ids[temp_ix_a]
    unit_b = p.unit_ids[temp_ix_b]
    spatial_a = p.spatial_singular[temp_ix_a]
    spatial_b = p.spatial_singular[temp_ix_b]

    # get shifted spatial components
    if p.is_drifting:
        spatial_a = drift_util.get_waveforms_on_static_channels(
            spatial_a,
            p.registered_geom,
            n_pitches_shift=shift_a,
            registered_geom=p.geom,
            target_kdtree=p.geom_kdtree,
            match_distance=p.match_distance,
            fill_value=0.0,
        )
        spatial_b = drift_util.get_waveforms_on_static_channels(
            spatial_b,
            p.registered_geom,
            n_pitches_shift=shift_b,
            registered_geom=p.geom,
            target_kdtree=p.geom_kdtree,
            match_distance=p.match_distance,
            fill_value=0.0,
        )

    # Explanation of all of the indexing going on below.
    #  - pair_ix_a,b index pairs of templates with nonzero cross-correlation
    #    so, there are i=1,...,N pairs pair_ix_a[i], pair_ix_b[i]
    #  - pair_ix is a N-array of indices such that...
    #  - up_pconv[pair_ix[i]] = set of upsampled pconv(pair_ix_a[i], pair_ix_b[i])
    #
    # these are normalized/deduplicated: pair_ix contains duplicate entries.
    # conv_ix contains the unique entries. in particular, the pconv between
    # pair_ix_a[i], pair_ix_b[i] is being computed as that between
    # pair_ix_a[conv_ix[pair_ix[i]]] and pair_ix_b[conv_ix[pair_ix[i]]].
    #
    # then, we also sparsify the temporal upsampling.

    # figure out pairs of templates to convolve
    pair_ix_a, pair_ix_b, pair_ix, conv_ix, shift_diff = deduplicated_pairs(
        shifted_temp_ix_a,
        shifted_temp_ix_b,
        spatial_a,
        spatial_b,
        temp_ix_a,
        temp_ix_b,
        shift_a=shift_a,
        shift_b=shift_b,
        cooccurrence=p.cooccurrence,
        conv_ignore_threshold=p.conv_ignore_threshold,
        geom=p.geom,
        registered_geom=p.registered_geom,
        reg_geom_kdtree=p.reg_geom_kdtree,
        match_distance=p.match_distance,
    )

    # to device
    spatial_a = spatial_a.to(p.device)
    spatial_b = spatial_b.to(p.device)
    temporal_a = p.temporal[temp_ix_a].to(p.device)
    temporal_up_b = p.temporal_up[temp_ix_b].to(p.device)

    # convolve valid pairs
    conv_ix_a, conv_ix_b, up_pconv, pair_survived = ccorrelate_up(
        spatial_a,
        temporal_a,
        spatial_b,
        temporal_up_b,
        conv_ignore_threshold=p.conv_ignore_threshold,
        max_shift=p.max_shift,
        pair_ix_a=pair_ix_a[conv_ix],
        pair_ix_b=pair_ix_b[conv_ix],
    )
    if conv_ix_a is None:
        return None
    nco = conv_ix_a.numel()
    if not nco:
        return None

    pair_ix = pair_ix[pair_survived]
    pair_ix_a = pair_ix_a[pair_survived]
    pair_ix_b = pair_ix_b[pair_survived]

    # # summarize units by coarse pconv when possible
    # if p.coarse_approx_error_threshold > 0:
    #     pconv, cconv_ix = _coarse_approx(
    #         pconv, cconv_ix, conv_ix_a, conv_ix_b, unit_a, unit_b, p
    #     )

    # for use in deconv residual distance merge
    # TODO: actually probably need to do the real objective here with
    # scaling. only need to do that bc of scaling right?
    # makes it kind of a pain, because then we need to go pairwise
    # (deconv objective is not symmetric)
    max_conv = best_shift = None
    if p.compute_max:
        cconv_ = pconv.reshape(nco, pconv.shape[1] * pconv.shape[2])
        max_conv, max_index = cconv_.max(dim=1)
        max_up, max_sample = np.unravel_index(
            max_index.numpy(force=True), shape=pconv.shape[1:]
        )
        best_shift = max_sample - (p.max_shift + 1)
        # if upsample>half nup, round max shift up
        best_shift += np.rint(max_up / pconv.shape[1]).astype(int)

    print(f"end {conv_ix_a.shape=}")
    print(f"end {conv_ix_b.shape=}")
    print(f"end {cconv_ix.shape=}")
    print(f"end {pconv.shape=}")

    return ConvBatchResult(
        shifted_temp_ix_a[pair_ix_a.numpy(force=True)],
        shifted_temp_ix_b[pair_ix_b.numpy(force=True)],
        cconv_ix,
        pconv.numpy(force=True) if pconv is not None else None,
        max_conv.numpy(force=True) if max_conv is not None else None,
        best_shift,
    )


# -- library code
# template index and shift pairs
# pairwise low-rank cross-correlation


@dataclass
class TemplateShiftIndex:
    """Return value for get_shift_and_unit_pairs"""

    n_shifted_templates: int
    # shift index -> shift
    all_pitch_shifts: np.ndarray
    # (template ix, shift index) -> shifted template index
    template_shift_index: np.ndarray
    # (shifted temp ix, shifted temp ix) -> did these appear at the same time
    cooccurrence: np.ndarray
    shifted_temp_ix_to_temp_ix: np.ndarray
    shifted_temp_ix_to_shift: np.ndarray


def static_template_shift_index(n_templates):
    temp_ixs = np.arange(n_templates)
    return TemplateShiftIndex(
        n_templates,
        np.zeros(1),
        temp_ixs[:, None],
        np.ones((n_templates, n_templates), dtype=bool),
        temp_ixs,
        np.zeros_like(temp_ixs),
    )


def get_shift_and_unit_pairs(
    chunk_time_centers_s,
    geom,
    template_data,
    motion_est=None,
):
    n_templates = len(template_data.templates)
    if motion_est is None:
        # no motion case
        return static_template_shift_index(n_templates)

    # all observed pitch shift values
    all_pitch_shifts = np.empty(shape=(0,), dtype=int)
    temp_ixs = np.arange(n_templates)
    # set of (template idx, shift)
    template_shift_pairs = np.empty(shape=(0, 2), dtype=int)
    pitch = drift_util.get_pitch(geom)

    for t_s in chunk_time_centers_s:
        # see the fn `templates_at_time`
        unregistered_depths_um = drift_util.invert_motion_estimate(
            motion_est, t_s, template_data.registered_template_depths_um
        )
        pitch_shifts = drift_util.get_spike_pitch_shifts(
            depths_um=template_data.registered_template_depths_um,
            pitch=pitch,
            registered_depths_um=unregistered_depths_um,
        )
        pitch_shifts = pitch_shifts.astype(int)

        # get unique pitch/unit shift pairs in chunk
        template_shift = np.c_[temp_ixs, pitch_shifts]

        # update full set
        all_pitch_shifts = np.union1d(all_pitch_shifts, pitch_shifts)
        template_shift_pairs = np.unique(
            np.concatenate((template_shift_pairs, template_shift), axis=0), axis=0
        )

    n_shifts = len(all_pitch_shifts)
    n_template_shift_pairs = len(template_shift_pairs)

    # index template/shift pairs: template_shift_index[template_ix, shift_ix] = shifted template index
    # fill with an invalid index
    template_shift_index = np.full((n_templates, n_shifts), n_template_shift_pairs)
    shift_ix = np.searchsorted(all_pitch_shifts, template_shift_pairs[:, 1])
    assert np.array_equal(all_pitch_shifts[shift_ix], template_shift_pairs[:, 1])
    template_shift_index[template_shift_pairs[:, 0], shift_ix] = np.arange(
        n_template_shift_pairs
    )
    shifted_temp_ix_to_temp_ix = template_shift_pairs[:, 0]
    shifted_temp_ix_to_shift = template_shift_pairs[:, 1]

    # co-occurrence matrix: do these shifted templates appear together?
    cooccurrence = np.eye(n_template_shift_pairs, dtype=bool)
    for t_s in chunk_time_centers_s:
        unregistered_depths_um = drift_util.invert_motion_estimate(
            motion_est, t_s, template_data.registered_template_depths_um
        )
        pitch_shifts = drift_util.get_spike_pitch_shifts(
            depths_um=template_data.registered_template_depths_um,
            pitch=pitch,
            registered_depths_um=unregistered_depths_um,
        )
        pitch_shifts = pitch_shifts.astype(int)
        pitch_shift_ix = np.searchsorted(all_pitch_shifts, pitch_shifts)

        shifted_temp_ixs = template_shift_index[temp_ixs, pitch_shift_ix]
        cooccurrence[shifted_temp_ixs[:, None], shifted_temp_ixs[None, :]] = 1

    return TemplateShiftIndex(
        n_template_shift_pairs,
        all_pitch_shifts,
        template_shift_index,
        cooccurrence,
        shifted_temp_ix_to_temp_ix,
        shifted_temp_ix_to_shift,
    )


def ccorrelate_up(
    spatial_a,
    temporal_a,
    spatial_b,
    temporal_b,
    upsampling_compression_map=None,
    conv_ignore_threshold=0.0,
    max_shift="full",
    covisible_mask=None,
    pair_ix_a=None,
    pair_ix_b=None,
    batch_size=128,
):
    """Convolve all pairs of low-rank templates

    This uses too much memory to run on all pairs at once.

    Templates Ka = Sa Ta, Kb = Sb Tb. The channel-summed convolution is
        (Ka (*) Kb) = sum_c Ka(c) * Kb(c)
                    = (Sb.T @ Ka) (*) Tb
                    = (Sb.T @ Sa @ Ta) (*) Tb
    where * is cross-correlation, and (*) is channel (or rank) summed.

    We use full-height conv2d to do rank-summed convs.

    upsampling_compression_map
        (n_templates, upsampling_factor)


    Returns
    -------
    covisible_a, covisible_b : tensors of indices
        Both have shape (nco,), where nco is the number of templates
        whose pairwise conv exceeds conv_ignore_threshold.
        So, zip(covisible_a, covisible_b) is the set of co-visible pairs.
    cconv : torch.Tensor
        Shape is (nco, nup, 2 * max_shift + 1)
        All cross-correlations for pairs of templates (templates in b
        can be upsampled.)
        If max_shift is full, then 2*max_shift+1=2t-1.
    """
    na, rank, nchan = spatial_a.shape
    nb, rank_, nchan_ = spatial_b.shape
    assert rank == rank_
    assert nchan == nchan_
    na_, t, rank_ = temporal_a.shape
    assert na == na_
    assert rank_ == rank
    nb_, t_, nup, rank_ = temporal_b.shape
    assert nb == nb_
    assert t == t_
    assert rank == rank_
    if covisible_mask is not None:
        assert covisible_mask.shape == (na, nb)

    # no need to convolve templates which do not overlap enough
    if pair_ix_a is None:
        covisible = (
            torch.sqrt(torch.square(spatial_a).sum(1))
            @ torch.sqrt(torch.square(spatial_b).sum(1)).T
        )
        covisible = covisible > conv_ignore_threshold
        if covisible_mask is not None:
            covisible *= covisible_mask
        covisible_a, covisible_b = torch.nonzero(covisible, as_tuple=True)
    else:
        covisible_a, covisible_b = pair_ix_a, pair_ix_b
    nco = covisible_a.numel()
    if not nco:
        return None, None, None

    # batch over nco for memory reasons
    cconv = torch.zeros(
        (nco, nup, 2 * max_shift + 1), dtype=spatial_a.dtype, device=spatial_a.device
    )
    for istart in range(0, nco, batch_size):
        iend = min(istart + batch_size, nco)
        co_a = covisible_a[istart:iend]
        co_b = covisible_b[istart:iend]
        nco_ = iend - istart

        # want conv filter: nco, 1, rank, t
        template_a = torch.bmm(temporal_a, spatial_a)
        conv_filt = torch.bmm(spatial_b[co_b], template_a[co_a].mT)
        conv_filt = conv_filt[:, None]  # (nco, 1, rank, t)

        # nup, nco, rank, t
        conv_in = temporal_b[co_b].permute(2, 0, 3, 1)

        # conv2d:
        # depthwise, chans=nco. batch=1. h=rank. w=t. out: nup, nco, 1, 2p+1.
        # input (conv_in): nup, nco, rank, t.
        # filters (conv_filt): nco, 1, rank, t. (groups=nco).
        cconv_ = F.conv2d(conv_in, conv_filt, padding=(0, max_shift), groups=nco_)
        cconv[istart:iend] = cconv_[:, :, 0, :].permute(1, 0, 2)  # nco, nup, time

    # more stringent covisibility
    pair_survived = slice(None)
    if conv_ignore_threshold > 0:
        max_val = cconv.reshape(nco, -1).abs().max(dim=1).values
        pair_survived = max_val > conv_ignore_threshold
        cconv = cconv[pair_survived]
        covisible_a = covisible_a[pair_survived]
        covisible_b = covisible_b[pair_survived]

    return covisible_a, covisible_b, cconv, pair_survived


# -- helpers


def deduplicated_pairs(
    shifted_temp_ix_a,
    shifted_temp_ix_b,
    spatialsing_a,
    spatialsing_b,
    temp_ix_a,
    temp_ix_b,
    shift_a=None,
    shift_b=None,
    cooccurrence=None,
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
    We don't need to convolve the same pair of templates twice, just where the indices
    are ordered (shifted_temp_ix_a <= shifted_temp_ix_b).

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
    dedup_ix
        Size == pair_ix_a,b size
        Subsets of conv_ix_a,b, so that the xcorr of templates
        shifted_temp_ix_a[pair_ix_a[i]], shifted_temp_ix_b[pair_ix_b[i]]
        is the same as that of
        shifted_temp_ix_a[conv_ix[dedup_ix[i]], conv_ix[dedup_ix[i]]]
    conv_ix
        Size < original number of shifted templates a,b
        Pairs of templates which should actually be convolved
    shift_diff
        Optional. if not None, same size as pair_ix_a
        shift_a - shift_b for this pair
    """
    # check spatially overlapping
    chan_amp_a = torch.sqrt(torch.square(spatialsing_a).sum(1))
    chan_amp_b = torch.sqrt(torch.square(spatialsing_b).sum(1))
    pair = chan_amp_a @ chan_amp_b.T
    pair = pair > conv_ignore_threshold

    # co-occurrence
    if cooccurrence is not None:
        pair *= cooccurrence

    # mask out lower triangle
    pair *= shifted_temp_ix_a[:, None] <= shifted_temp_ix_b[None, :]
    pair_ix_a, pair_ix_b = torch.nonzero(pair, as_tuple=True)
    nco = pair_ix_a.numel()

    # if no shifting, deduplication is the identity
    if shift_a is None:
        assert shift_b is None
        nco_range = torch.arange(nco, device=pair_ix_a.device)
        return pair_ix_a, pair_ix_b, nco_range, nco_range, None

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
    _, active_chan_ids_a = np.unique(active_chans_a, axis=0, return_inverse=True)
    _, active_chan_ids_b = np.unique(active_chans_b, axis=0, return_inverse=True)

    # 3
    temp_ix_a = temp_ix_a[pair_ix_a]
    temp_ix_b = temp_ix_b[pair_ix_b]
    # get the relative shifts
    shift_a = shift_a[pair_ix_a]
    shift_b = shift_b[pair_ix_b]
    shift_diff = shift_a - shift_b

    # figure out combinations
    conv_determiners = np.c_[
        temp_ix_a,
        active_chan_ids_a[pair_ix_a],
        temp_ix_b,
        active_chan_ids_b[pair_ix_b],
        shift_diff,
    ]
    # conv_ix: indices of unique determiners
    # dedup_ix: which representative does each pair belong to
    _, conv_ix, dedup_ix = np.unique(
        conv_determiners, axis=0, return_index=True, return_inverse=True
    )

    return pair_ix_a, pair_ix_b, dedup_ix, conv_ix, shift_diff



def _coarse_approx(cconv, cconv_ix, conv_ix_a, conv_ix_b, unit_a, unit_b, p):
    # figure out coarse templates to correlate
    conv_ix_a = conv_ix_a.cpu()
    conv_ix_b = conv_ix_b.cpu()
    conv_unit_a = unit_a[conv_ix_a]
    conv_unit_b = unit_b[conv_ix_b]
    coarse_units_a = np.unique(conv_unit_a)
    coarse_units_b = np.unique(conv_unit_b)
    coarsecovis = np.zeros((coarse_units_a.size, coarse_units_b.size), dtype=bool)
    coarsecovis[
        np.searchsorted(coarse_units_a, conv_unit_a),
        np.searchsorted(coarse_units_b, conv_unit_b),
    ] = True

    # correlate them
    coarse_ix_a, coarse_ix_b, coarse_cconv = ccorrelate_up(
        p.coarse_spatial_singular[coarse_units_a].to(p.device),
        p.coarse_temporal[coarse_units_a].to(p.device),
        p.coarse_spatial_singular[coarse_units_b].to(p.device),
        p.coarse_temporal[coarse_units_b].unsqueeze(2).to(p.device),
        conv_ignore_threshold=p.conv_ignore_threshold,
        max_shift=p.max_shift,
        covisible_mask=torch.as_tensor(coarsecovis, device=p.device),
    )
    if coarse_ix_a is None:
        return cconv, cconv_ix

    coarse_units_a = np.atleast_1d(coarse_units_a[coarse_ix_a.cpu()])
    coarse_units_b = np.atleast_1d(coarse_units_b[coarse_ix_b.cpu()])

    # find coarse units which well summarize the fine cconvs
    for coarse_unit_a, coarse_unit_b, conv in zip(
        coarse_units_a, coarse_units_b, coarse_cconv
    ):
        # check good approx. if not, continue
        in_pair = np.flatnonzero(
            (conv_unit_a == coarse_unit_a) & (conv_unit_b == coarse_unit_b)
        )
        assert in_pair.size
        fine_cconvs = cconv[cconv_ix[in_pair]]
        approx_err = (fine_cconvs - conv[None]).abs().max()
        if not approx_err < p.coarse_approx_error_threshold:
            continue

        # replace first fine cconv with the coarse cconv
        cconv[cconv_ix[in_pair[0]]] = conv
        # set all fine cconv ix to the index of that first one
        cconv_ix[in_pair] = cconv_ix[in_pair[0]]

    # re-index and subset cconvs
    cconv_ix_subset, new_cconv_ix = np.unique(cconv_ix, return_inverse=True)
    cconv = cconv[cconv_ix_subset]
    return cconv, new_cconv_ix
