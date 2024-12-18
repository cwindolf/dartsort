from dataclasses import dataclass, fields
from typing import Optional

import h5py
import numpy as np
import torch

from .pairwise_util import compressed_convolve_to_h5
from .template_util import CompressedUpsampledTemplates, LowRankTemplates
from .templates import TemplateData
from ..util.data_util import batched_h5_read


@dataclass
class CompressedPairwiseConv:
    """A database of channel-summed cross-correlations between template pairs

    There are too many templates to store all of these, especially after
    superres binning, temporal upsampling, and pitch shifting. We compress
    this as much as possible, first by deduplication (many convolutions of
    templates at different shifts are identical), next by not wasting space
    (no need to compute as many upsampled copies of small templates), and
    finally by approximation (for pairs of far-away units, correlations of
    superres templates are very close to correlations of the non-superres
    template).

    This database holds some indexing structures that help us store these
    correlations sparsely. .query() grabs the actual correlations for the
    user.
    """

    # shape: (n_shifts,)
    # shift_ix -> shift (pitch shift, an integer)
    shifts_a: np.ndarray
    shifts_b: np.ndarray

    # shape: (n_templates_a, n_shifts_a)
    # (template_ix, shift_ix) -> shifted_template_ix
    # shifted_template_ix can be either invalid (this template does not occur
    # at this shift), or it can range from 0, ..., n_shifted_templates_a-1
    shifted_template_index_a: np.ndarray

    # shape: (n_templates_b, n_shifts_b, upsampling_factor)
    # (template_ix, shift_ix, upsampling_ix) -> upsampled_shifted_template_ix
    upsampled_shifted_template_index_b: np.ndarray

    # shape: (n_shifted_templates_a, n_upsampled_shifted_templates_b)
    # (shifted_template_ix, upsampled_shifted_template_ix) -> pconv_ix
    pconv_index: np.ndarray

    # shape: (n_pconvs, 2 * spike_length_samples - 1)
    # pconv_ix -> a cross-correlation array
    # the 0 index is special: pconv[0] === 0.
    pconv: np.ndarray
    in_memory: bool = False
    device: torch.device = torch.device("cpu")

    def __post_init__(self):
        assert self.shifts_a.ndim == self.shifts_b.ndim == 1
        assert self.shifts_a.shape == (self.shifted_template_index_a.shape[1],)
        assert self.shifts_b.shape == (
            self.upsampled_shifted_template_index_b.shape[1],
        )
        self.a_shift_offset, self.offset_shift_a_to_ix = _get_shift_indexer(
            self.shifts_a
        )
        self.b_shift_offset, self.offset_shift_b_to_ix = _get_shift_indexer(
            self.shifts_b
        )

    def get_shift_ix_a(self, shifts_a):
        """Map shift (an integer, signed) to a shift index

        A shift index can be used to index into axis=1 of shifted_template_index_a,
        or self.shifts_a for that matter.
        It's an int in [0, n_shifts_a).
        It's equal to np.searchsorted(self.shifts_a, shifts_a).
        The thing is, searchsorted is slow, and we can pre-bake a lookup table.
        _get_shift_indexer does the baking for us below.
        """
        shifts_a = torch.atleast_1d(torch.as_tensor(shifts_a))
        return self.offset_shift_a_to_ix[shifts_a.to(int) + self.a_shift_offset]

    def get_shift_ix_b(self, shifts_b):
        shifts_b = torch.atleast_1d(torch.as_tensor(shifts_b))
        return self.offset_shift_b_to_ix[shifts_b.to(int) + self.b_shift_offset]

    @classmethod
    def from_h5(cls, hdf5_filename, in_memory=True):
        ff = [f for f in fields(cls) if f.name not in ("in_memory", "device")]
        if in_memory:
            with h5py.File(hdf5_filename, "r") as h5:
                data = {f.name: torch.from_numpy(h5[f.name][:]) for f in ff}
            return cls(**data, in_memory=in_memory)
        _h5 = h5py.File(hdf5_filename, "r")
        data = {}
        for f in ff:
            if f.name == "pconv":
                data[f.name] = _h5[f.name]
            else:
                data[f.name] = torch.from_numpy(_h5[f.name][:])
        return cls(**data, in_memory=in_memory)

    @classmethod
    def from_template_data(
        cls,
        hdf5_filename,
        template_data: TemplateData,
        low_rank_templates: LowRankTemplates,
        compressed_upsampled_temporal: CompressedUpsampledTemplates,
        template_data_b: Optional[TemplateData] = None,
        low_rank_templates_b: Optional[TemplateData] = None,
        chunk_time_centers_s: Optional[np.ndarray] = None,
        motion_est=None,
        geom: Optional[np.ndarray] = None,
        conv_ignore_threshold=0.0,
        coarse_approx_error_threshold=0.0,
        conv_batch_size=1024,
        units_batch_size=8,
        overwrite=False,
        computation_config=None,
        show_progress=True,
    ):
        if computation_config is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            n_jobs = 0
        else:
            n_jobs = computation_config.actual_n_jobs()
            device = computation_config.device()

        compressed_convolve_to_h5(
            hdf5_filename,
            template_data=template_data,
            low_rank_templates=low_rank_templates,
            compressed_upsampled_temporal=compressed_upsampled_temporal,
            template_data_b=template_data_b,
            low_rank_templates_b=low_rank_templates_b,
            chunk_time_centers_s=chunk_time_centers_s,
            motion_est=motion_est,
            geom=geom,
            conv_ignore_threshold=conv_ignore_threshold,
            coarse_approx_error_threshold=coarse_approx_error_threshold,
            conv_batch_size=conv_batch_size,
            units_batch_size=units_batch_size,
            overwrite=overwrite,
            device=device,
            n_jobs=n_jobs,
            show_progress=show_progress,
        )
        return cls.from_h5(hdf5_filename)

    def to(self, device=None, incl_pconv=False, pin=False):
        """Become torch tensors on device."""
        for name in ["offset_shift_a_to_ix", "offset_shift_b_to_ix"] + [
            f.name for f in fields(self)
        ]:
            if name == "pconv" and not incl_pconv:
                continue
            v = getattr(self, name)
            if isinstance(v, np.ndarray) or torch.is_tensor(v):
                setattr(self, name, torch.as_tensor(v, device=device))
        self.device = device
        if (
            pin
            and self.device.type == "cuda"
            and torch.cuda.is_available()
            and not self.pconv.is_pinned()
        ):
            # self.pconv.share_memory_()
            print("pin")
            torch.cuda.cudart().cudaHostRegister(
                self.pconv.data_ptr(), self.pconv.numel() * self.pconv.element_size(), 0
            )
            # assert x.is_shared()
            assert self.pconv.is_pinned()
            # self.pconv = self.pconv.pin_memory()
        return self

    def query(
        self,
        template_indices_a,
        template_indices_b,
        upsampling_indices_b=None,
        shifts_a=None,
        shifts_b=None,
        scalings_b=None,
        times_b=None,
        return_zero_convs=False,
        grid=False,
        device=None,
    ):
        if template_indices_a is None:
            template_indices_a = torch.arange(
                len(self.shifted_template_index_a), device=self.device
            )
        template_indices_a = torch.atleast_1d(torch.as_tensor(template_indices_a))
        template_indices_b = torch.atleast_1d(torch.as_tensor(template_indices_b))

        # handle no shifting
        no_shifting = shifts_a is None or shifts_b is None
        shifted_template_index = self.shifted_template_index_a
        upsampled_shifted_template_index = self.upsampled_shifted_template_index_b
        if no_shifting:
            assert shifts_a is None and shifts_b is None
            assert self.shifts_a.shape == (1,)
            assert self.shifts_b.shape == (1,)
            a_ix = (template_indices_a,)
            b_ix = (template_indices_b,)
            shifted_template_index = shifted_template_index[:, 0]
            upsampled_shifted_template_index = upsampled_shifted_template_index[:, 0]
        else:
            shift_indices_a = self.get_shift_ix_a(shifts_a)
            shift_indices_b = self.get_shift_ix_a(shifts_b)
            a_ix = (template_indices_a, shift_indices_a)
            b_ix = (template_indices_b, shift_indices_b)

        # handle no upsampling
        no_upsampling = upsampling_indices_b is None
        if no_upsampling:
            assert self.upsampled_shifted_template_index_b.shape[2] == 1
            upsampled_shifted_template_index = upsampled_shifted_template_index[..., 0]
        else:
            b_ix = b_ix + (torch.atleast_1d(torch.as_tensor(upsampling_indices_b)),)

        # get shifted template indices for A
        shifted_temp_ix_a = shifted_template_index[a_ix]

        # upsampled shifted template indices for B
        up_shifted_temp_ix_b = upsampled_shifted_template_index[b_ix]

        # return convolutions between all ai,bj or just ai,bi?
        if grid:
            pconv_indices = self.pconv_index[
                shifted_temp_ix_a[:, None], up_shifted_temp_ix_b[None, :]
            ]
            template_indices_a, template_indices_b = torch.cartesian_prod(
                template_indices_a, template_indices_b
            ).T
            if scalings_b is not None:
                scalings_b = torch.broadcast_to(
                    scalings_b[None], pconv_indices.shape
                ).reshape(-1)
            if times_b is not None:
                times_b = torch.broadcast_to(
                    times_b[None], pconv_indices.shape
                ).reshape(-1)
            pconv_indices = pconv_indices.view(-1)
        else:
            pconv_indices = self.pconv_index[shifted_temp_ix_a, up_shifted_temp_ix_b]

        # most users will be happy not to get a bunch of zeros for pairs that don't overlap
        if not return_zero_convs:
            which = pconv_indices > 0
            pconv_indices = pconv_indices[which]
            template_indices_a = template_indices_a[which]
            template_indices_b = template_indices_b[which]
            if scalings_b is not None:
                scalings_b = scalings_b[which]
            if times_b is not None:
                times_b = times_b[which]

        if self.in_memory:
            pconvs = self.pconv[pconv_indices.to(self.pconv.device)]
        else:
            pconvs = torch.from_numpy(
                batched_h5_read(self.pconv, pconv_indices.numpy(force=True))
            )
        if device is not None:
            pconvs = pconvs.to(device)

        if scalings_b is not None:
            pconvs.mul_(scalings_b[:, None])

        if times_b is not None:
            return template_indices_a, template_indices_b, times_b, pconvs

        return template_indices_a, template_indices_b, pconvs

    # def at_shifts(self, shifts_a=None, shifts_b=None, device=None):
    #     """Subset this database to one set of shifts.

    #     The database becomes shiftless (not in the pejorative sense).
    #     """
    #     if shifts_a is None or shifts_b is None:
    #         assert shifts_a is shifts_b
    #         assert self.shifts_a.shape == (1,)
    #         assert self.shifts_b.shape == (1,)
    #         return self

    #     assert shifts_a.shape == (len(self.shifted_template_index_a),)
    #     assert shifts_b.shape == (len(self.upsampled_shifted_template_index_b),)
    #     n_shifted_temps_a, n_up_shifted_temps_b = self.pconv_index.shape

    #     # active shifted and upsampled indices
    #     shift_ix_a = self.get_shift_ix_a(shifts_a)
    #     shift_ix_b = self.get_shift_ix_b(shifts_b)
    #     sub_shifted_temp_index_a = self.shifted_template_index_a[
    #         torch.arange(len(self.shifted_template_index_a))[:, None],
    #         shift_ix_a[:, None],
    #     ]
    #     sub_up_shifted_temp_index_b = self.upsampled_shifted_template_index_b[
    #         torch.arange(len(self.upsampled_shifted_template_index_b))[:, None],
    #         shift_ix_b[:, None],
    #     ]

    #     # in flat form for indexing into pconv_index. also, reindex.
    #     valid_a = sub_shifted_temp_index_a < n_shifted_temps_a
    #     shifted_temp_ixs_a, new_shifted_temp_ixs_a = torch.unique(
    #         sub_shifted_temp_index_a[valid_a], return_inverse=True
    #     )
    #     valid_b = sub_up_shifted_temp_index_b < n_up_shifted_temps_b
    #     up_shifted_temp_ixs_b, new_up_shifted_temp_ixs_b = torch.unique(
    #         sub_up_shifted_temp_index_b[valid_b], return_inverse=True
    #     )

    #     # get relevant pconv subset and reindex
    #     sub_pconv_indices, new_pconv_indices = torch.unique(
    #         self.pconv_index[
    #             shifted_temp_ixs_a[:, None],
    #             up_shifted_temp_ixs_b.ravel()[None, :],
    #         ],
    #         return_inverse=True,
    #     )
    #     if self.in_memory:
    #         sub_pconv = self.pconv[sub_pconv_indices.to(self.pconv.device)]
    #     else:
    #         sub_pconv = torch.from_numpy(batched_h5_read(self.pconv, sub_pconv_indices))
    #     if device is not None:
    #         sub_pconv = sub_pconv.to(device)

    #     # reindexing
    #     n_sub_shifted_temps_a = len(shifted_temp_ixs_a)
    #     n_sub_up_shifted_temps_b = len(up_shifted_temp_ixs_b)
    #     sub_pconv_index = new_pconv_indices.view(
    #         n_sub_shifted_temps_a, n_sub_up_shifted_temps_b
    #     )
    #     sub_shifted_temp_index_a[valid_a] = new_shifted_temp_ixs_a
    #     sub_up_shifted_temp_index_b[valid_b] = new_up_shifted_temp_ixs_b

    #     return self.__class__(
    #         shifts_a=torch.zeros(1),
    #         shifts_b=torch.zeros(1),
    #         shifted_template_index_a=sub_shifted_temp_index_a,
    #         upsampled_shifted_template_index_b=sub_up_shifted_temp_index_b,
    #         pconv_index=sub_pconv_index,
    #         pconv=sub_pconv,
    #         in_memory=True,
    #         device=self.device,
    #     )


def _get_shift_indexer(shifts):
    assert torch.equal(shifts, torch.sort(shifts).values)
    # smallest shift (say, -5) becomes 5
    shift_offset = -int(shifts[0])
    offset_shift_to_ix = []

    for j, shift in enumerate(shifts):
        ix = shift + shift_offset
        assert len(offset_shift_to_ix) <= ix

        # fill indices corresponding to missing shifts with an out-of-bounds
        # index to cause a panic if someone tries to load up a shift which DNE
        while len(offset_shift_to_ix) < ix:
            offset_shift_to_ix.append(len(shifts))

        # real shifts get good index
        offset_shift_to_ix.append(j)

    offset_shift_to_ix = torch.tensor(offset_shift_to_ix, device=shifts.device)
    return shift_offset, offset_shift_to_ix
