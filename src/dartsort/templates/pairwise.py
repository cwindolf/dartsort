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
    is_drifting = np.array_equal(temp_shift_index.all_pitch_shifts, [0])
    if template_data.registered_geom is not None:
        is_drifting &= np.array_equal(geom, template_data.registered_geom)

    # initialize pairwise conv data structures
    # index_table[shifted_temp_ix(i), shifted_temp_ix(j)] = pconvix(i,j)
    pconv_index_table = np.zeros(
        (temp_shift_index.n_shifted_templates, temp_shift_index.n_shifted_templates),
        dtype=int,
    )
    # pconvs[pconvix(i,j)] = (2*spikelen-1, upsampling_factor) arr of pconv(shifted_temp(i), shifted_temp(j))

    cur_pconv_ix = 1
    with h5py.File(output_hdf5_filename, "w") as h5:
        # resizeable pconv dataset
        pconv = h5.create_dataset(
            "pconv",
            dtype=np.float32,
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
            new_conv_ix = res.cconv_ix
            ixnz = new_conv_ix > 0
            new_conv_ix[ixnz] += cur_pconv_ix
            pconv_index_table[
                res.shifted_temp_ix_a, res.shifted_temp_ix_b
            ] = new_conv_ix
            pconv.resize(cur_pconv_ix + ixnz.size, axis=0)
            pconv[new_conv_ix[ixnz]] = res.pconv[ixnz]
            cur_pconv_ix += ixnz.size

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

    return SparsePairwiseConv.from_h5(output_hdf5_filename)


@dataclass
class SparsePairwiseConv:
    # shift_ix -> shift
    shifts: np.ndarray
    # (temp_ix, shift_ix) -> shifted_temp_ix
    template_shift_index: torch.LongTensor
    # (shifted_temp_ix a, shifted_temp_ix b) -> pconv index
    pconv_index_table: torch.LongTensor
    # pconv index -> pconv (upsampling, 2 * spike len - 1)
    # the zero index lands you at an all 0 pconv
    pconv: torch.Tensor

    # metadata: map shifted template index to original template ix and shift
    shifted_temp_ix_to_temp_ix: np.ndarray
    shifted_temp_ix_to_shift: np.ndarray
    shifted_temp_ix_to_unit: np.ndarray

    @classmethod
    def from_h5(cls, hdf5_filename):
        ff = fields(cls)
        with h5py.File(hdf5_filename, "r") as h5:
            data = {f.name: h5[f.name][:] for f in ff}
        return cls(**data)

    def query(
        self,
        template_indices_a,
        template_indices_b,
        upsampling_indices_b=None,
        shifts_a=None,
        shifts_b=None,
        return_zero_convs=False,
    ):
        """Get cross-correlations of pairs of units A and B

        This passes through the series of lookup tables to recover (upsampled)
        cross-correlations from this sparse database.

        Returns
        -------
        template_indices_a, template_indices_b, pair_convs
        """
        # get shifted template indices
        pconv = self.pconv
        if upsampling_indices_b is None:
            assert self.pconv.shape[1] == 1
            pconv = pconv[:, 0, :]
        if shifts_a is None or shifts_b is None:
            assert np.array_equal(self.shifts, [0.0])
            shifted_temp_ix_a = template_indices_a
            shifted_temp_ix_b = template_indices_b
        else:
            shift_ix_a = np.searchsorted(self.shifts, shifts_a)
            assert np.array_equal(self.shifts[shift_ix_a], shifts_a)
            shift_ix_b = np.searchsorted(self.shifts, shifts_b)
            assert np.array_equal(self.shifts[shift_ix_b], shifts_b)
            shifted_temp_ix_a = self.template_shift_index[
                template_indices_a, shift_ix_a
            ]
            shifted_temp_ix_b = self.template_shift_index[
                template_indices_b, shift_ix_b
            ]

        pconv_indices = self.pconv_index_table[shifted_temp_ix_a, shifted_temp_ix_b]

        # most users will be happy not to get a bunch of zeros for pairs that don't overlap
        if not return_zero_convs:
            which = np.flatnonzero(pconv_indices > 0)
            pconv_indices = pconv_indices[which]
            template_indices_a = template_indices_a[which]
            template_indices_b = template_indices_b[which]
            if upsampling_indices_b is not None:
                upsampling_indices_b = upsampling_indices_b[which]

        if upsampling_indices_b is None:
            pair_convs = pconv[pconv_indices]
        else:
            pair_convs = pconv[pconv_indices, upsampling_indices_b]

        return template_indices_a, template_indices_b, pair_convs


def compute_pairwise_convs(
    template_data,
    spatial,
    singular,
    temporal,
    temporal_up,
    shifted_temp_ix_to_temp_ix,
    shifted_temp_ix_to_shift,
    geom,
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
        for start_b in range(start_a + 1, units.size, units_per_chunk):
            end_b = min(start_b + units_per_chunk, units.size)
            jobs.append((units[start_a:end_a], units[start_b:end_b]))
    if show_progress:
        jobs = tqdm(jobs)

    # compute the coarse templates if needed
    if units.size == template_data.unit_ids.size:
        # coarse templates are original templates
        coarse_approx_error_threshold = 0
    if coarse_approx_error_threshold > 0:
        coarse_templates = template_util.weighted_average(
            template_data.unit_ids, template_data.templates, template_data.spike_counts
        )
        (
            coarse_spatial,
            coarse_singular,
            coarse_temporal,
        ) = template_util.svd_compress_templates(
            coarse_templates,
            rank=spatial.shape[2],
            min_channel_amplitude=min_channel_amplitude,
        )

    # template data to torch
    spatial_singular = torch.as_tensor(spatial * singular[:, None, :])
    temporal = torch.as_tensor(temporal)
    temporal_up = torch.as_tensor(temporal_up)
    if coarse_approx_error_threshold > 0:
        coarse_spatial_singular = torch.as_tensor(
            coarse_spatial * coarse_singular[:, None, :]
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

    # template indexing helper arrays
    unit_ids: np.ndarray
    shifted_temp_ix_to_temp_ix: np.ndarray
    shifted_temp_ix_to_shift: np.ndarray

    # only needed if is_drifting
    geom: np.ndarray
    registered_geom: np.ndarray
    target_kdtree: Optional[KDTree]
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

    kwargs["target_kdtree"] = kwargs["match_distance"] = None
    if kwargs["is_drifting"]:
        kwargs["target_kdtree"] = KDTree(kwargs["geom"])
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


def _pairwise_conv_job(
    units_a,
    units_b,
):
    """units_a,b are chunks of original (non-superres) unit labels"""
    global _pairwise_conv_context
    p = _pairwise_conv_context

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

    # get shifted spatial components
    spatial_a = p.spatial_singular[temp_ix_a]
    spatial_b = p.spatial_singular[temp_ix_b]
    if p.is_drifting:
        spatial_a = drift_util.get_waveforms_on_static_channels(
            spatial_a,
            p.registered_geom,
            n_pitches_shift=shift_a,
            registered_geom=p.geom,
            target_kdtree=p.target_kdtree,
            match_distance=p.match_distance,
            fill_value=0.0,
        )
        spatial_b = drift_util.get_waveforms_on_static_channels(
            spatial_b,
            p.registered_geom,
            n_pitches_shift=shift_b,
            registered_geom=p.geom,
            target_kdtree=p.target_kdtree,
            match_distance=p.match_distance,
            fill_value=0.0,
        )

    # to device
    spatial_a = spatial_a.to(p.device)
    spatial_b = spatial_b.to(p.device)
    temporal_a = p.temporal[temp_ix_a].to(p.device)
    temporal_up_b = p.temporal_up[temp_ix_b].to(p.device)

    # convolve valid pairs
    pair_mask = p.cooccurence[shifted_temp_ix_a[:, None], shifted_temp_ix_b[None, :]]
    pair_mask = pair_mask * (shifted_temp_ix_a[:, None] <= shifted_temp_ix_b[None, :])
    pair_mask = torch.as_tensor(pair_mask, device=p.device)
    conv_ix_a, conv_ix_b, cconv = ccorrelate_up(
        spatial_a,
        temporal_a,
        spatial_b,
        temporal_up_b,
        conv_ignore_threshold=p.conv_ignore_threshold,
        max_shift=p.max_shift,
        covisible_mask=pair_mask,
    )
    nco = conv_ix_a.numel()
    cconv_ix = np.arange(nco)

    # summarize units by coarse pconv when possible
    if p.coarse_approx_error_threshold > 0:
        # figure out coarse templates to correlate
        conv_unit_a = unit_a[conv_ix_a]
        conv_unit_b = unit_b[conv_ix_b]
        coarse_units_a, coarse_units_b = np.unique(
            np.c_[conv_unit_a, conv_unit_b],
            axis=0,
        ).T

        # correlate them
        coarse_ix_a, coarse_ix_b, coarse_cconv = ccorrelate_up(
            p.coarse_spatial_singular[coarse_units_a],
            p.coarse_temporal[temp_ix_a],
            p.coarse_spatial_singular[coarse_units_b],
            p.coarse_temporal[temp_ix_b].unsqueeze(2),
            conv_ignore_threshold=p.conv_ignore_threshold,
            max_shift=p.max_shift,
        )
        # i feel like this should hold so assert for now
        assert coarse_ix_a.size == coarse_units_a.size

        # find coarse units which well summarize the fine cconvs
        for coarse_unit_a, coarse_unit_b, conv in zip(
            coarse_units_a, coarse_units_b, coarse_cconv
        ):
            # check good approx. if not, continue
            in_pair = np.flatnonzero(
                (conv_unit_a == coarse_unit_a) & (conv_unit_b == coarse_unit_b)
            )
            assert in_pair.size
            fine_cconvs = cconv[in_pair]
            approx_err = (fine_cconvs - conv[None]).abs().max()
            if not approx_err < p.coarse_approx_error_threshold:
                continue

            # replace first fine cconv with the coarse cconv
            fine_cconvs[in_pair[0]] = conv
            # set all fine cconv ix to the index of that first one
            cconv_ix[in_pair] = cconv_ix[in_pair[0]]

        # re-index and subset cconvs
        cconv_ix = np.unique(cconv_ix)
        conv_ix_a = conv_ix_a[cconv_ix]
        conv_ix_b = conv_ix_b[cconv_ix]
        cconv = cconv[cconv_ix]

    # for use in deconv residual distance merge
    # TODO: actually probably need to do the real objective here with
    # scaling. only need to do that bc of scaling right?
    # makes it kind of a pain, because then we need to go pairwise
    # (deconv objective is not symmetric)
    max_conv = best_shift = None
    if p.compute_max:
        cconv_ = cconv.reshape(nco, cconv.shape[1] * cconv.shape[2])
        max_conv, max_index = cconv_.max(dim=1)
        max_up, max_sample = np.unravel_index(
            max_index.numpy(force=True), shape=cconv.shape[1:]
        )
        best_shift = max_sample - (p.max_shift + 1)
        # if upsample>half nup, round max shift up
        best_shift += np.rint(max_up / cconv.shape[1]).astype(int)

    return ConvBatchResult(
        shifted_temp_ix_a[conv_ix_a].numpy(force=True),
        shifted_temp_ix_b[conv_ix_b].numpy(force=True),
        cconv_ix,
        cconv.numpy(force=True),
        max_conv.numpy(force=True),
        best_shift,
    )


# -- library code
# template index and shift pairs
# pairwise low-rank cross-correlation


# this dtype lets us use np.union1d to find unique
# template index + pitch shift pairs below
template_shift_pair = np.dtype([("template_ix", int), ("shift", int)])


@dataclass
class TemplateShiftIndex:
    """Return value for get_shift_and_unit_pairs"""

    n_shifted_templates: int
    # shift index -> shift
    all_pitch_shifts: np.ndarray
    # (template ix, shift index) -> shifted template index
    template_shift_index: np.ndarray
    # (shifted temp ix, shifted temp ix) -> did these appear at the same time
    cooccurence: np.ndarray
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
    all_pitch_shifts = np.empty(shape=(), dtype=int)
    temp_ixs = np.arange(n_templates)
    # set of (template idx, shift)
    template_shift_pairs = np.empty(shape=(), dtype=template_shift_pair)

    for t_s in chunk_time_centers_s:
        # see the fn `templates_at_time`
        unregistered_depths_um = drift_util.invert_motion_estimate(
            motion_est, t_s, template_data.registered_template_depths_um
        )
        pitch_shifts = drift_util.get_spike_pitch_shifts(
            depths_um=template_data.registered_template_depths_um,
            geom=geom,
            registered_depths_um=unregistered_depths_um,
        )
        pitch_shifts = pitch_shifts.astype(int)

        # get unique pitch/unit shift pairs in chunk
        template_shift = np.c_[temp_ixs, pitch_shifts]
        template_shift = template_shift.view(template_shift_pair)[:, 0]
        assert template_shift.shape == (n_templates,)

        # update full set
        all_pitch_shifts = np.union1d(all_pitch_shifts, pitch_shifts)
        template_shift_pairs = np.union1d(template_shift_pairs, template_shift)

    n_shifts = len(all_pitch_shifts)
    n_template_shift_pairs = len(template_shift_pairs)

    # index template/shift pairs: template_shift_index[template_ix, shift_ix] = shifted template index
    # fill with an invalid index
    template_shift_index = np.full((n_templates, n_shifts), n_template_shift_pairs + 1)
    template_shift_index[
        template_shift_pairs["template_ix"], template_shift_pairs["shift"]
    ] = np.arange(n_template_shift_pairs)
    shifted_temp_ix_to_temp_ix = template_shift_pairs["template_ix"]
    shifted_temp_ix_to_shift = template_shift_pairs["shift"]

    # co-occurrence matrix: do these shifted templates appear together?
    cooccurence = np.zeros((n_template_shift_pairs, n_template_shift_pairs), dtype=bool)
    for t_s in chunk_time_centers_s:
        # see the fn `templates_at_time`
        unregistered_depths_um = drift_util.invert_motion_estimate(
            motion_est, t_s, template_data.registered_template_depths_um
        )
        pitch_shifts = drift_util.get_spike_pitch_shifts(
            depths_um=template_data.registered_template_depths_um,
            geom=geom,
            registered_depths_um=unregistered_depths_um,
        )
        pitch_shifts = pitch_shifts.astype(int)

        shifted_temp_ixs = template_shift_index[temp_ixs, pitch_shifts]
        cooccurence[shifted_temp_ixs[:, None], shifted_temp_ixs[None, :]] = 1

    return TemplateShiftIndex(
        n_template_shift_pairs,
        all_pitch_shifts,
        template_shift_index,
        cooccurence,
        shifted_temp_ix_to_temp_ix,
        shifted_temp_ix_to_shift,
    )


def ccorrelate_up(
    spatial_a,
    temporal_a,
    spatial_b,
    temporal_b,
    conv_ignore_threshold=0.0,
    max_shift="full",
    covisible_mask=None,
):
    """Convolve all pairs of low-rank templates

    This uses too much memory to run on all pairs at once.

    Templates Ka = Sa Ta, Kb = Sb Tb. The channel-summed convolution is
        (Ka (*) Kb) = sum_c Ka(c) * Kb(c)
                    = (Sb.T @ Ka) (*) Tb
                    = (Sb.T @ Sa @ Ta) (*) Tb
    where * is cross-correlation, and (*) is channel (or rank) summed.

    We use full-height conv2d to do rank-summed convs.

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

    # this is covisible with ignore threshold 0
    # no need to convolve templates which do not overlap
    covisible = spatial_a.max(1).values @ spatial_b.max(1).values.T
    if covisible_mask is not None:
        covisible *= covisible_mask
    covisible_a, covisible_b = torch.nonzero(covisible, as_tuple=True)
    nco = covisible_a.numel()
    # TODO: can batch over nco dims below if memory issues arise

    Sa = spatial_a[covisible_a].reshape(nco * rank, nchan)
    Sb = spatial_b[covisible_b].reshape(nco * rank, nchan)
    spatial_outer = torch.vecdot(Sa, Sb)
    spatial_outer = spatial_outer.reshape(nco, rank)
    assert spatial_outer.shape == (nco, rank)

    # want conv filter: nco, rank, t
    spatial_outer_co = spatial_outer[covisible_a, covisible_b]
    conv_filt = spatial_outer_co[:, None, :] * temporal_a.permute(0, 2, 1)[None]
    assert conv_filt.shape == (nco, rank, t)

    # nup, nco, rank, t
    conv_in = temporal_b[covisible_b].permute(2, 0, 3, 1)

    # conv2d:
    # depthwise, chans=nco. batch=1. h=rank. w=t. out: nup, nco, 1, 2p+1.
    # input (conv_left): nup, nco, rank, t.
    # filters (conv_right): nco, 1, rank, t. (groups=nco).
    cconv = F.conv2d(conv_in, conv_filt, padding=max_shift, groups=nco)
    assert cconv.shape == (nup, nco, 1, 2 * max_shift + 1)
    cconv = cconv[:, :, 0, :].permute(1, 0, 2)

    # more stringent covisibility
    if conv_ignore_threshold > 0:
        vis = cconv.abs().max(dim=(0, 2)).values > conv_ignore_threshold
        cconv = cconv[vis]
        covisible_a = covisible_a[vis]
        covisible_b = covisible_b[vis]

    return covisible_a, covisible_b, cconv
