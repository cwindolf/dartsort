from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from ...templates.template_util import CompressedUpsampledTemplates, LowRankTemplates
from ...templates.templates import TemplateData
from ...util.py_util import resolve_path
from ...util import job_util
from .pairwise_util import compressed_convolve_to_h5
from .matching_base import PconvBase


class CompressedPairwiseConv(PconvBase):
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

    # shifts_b
    # shape: (n_shifts,)
    # shift_ix -> shift (pitch shift, an integer)

    # shifted_template_index_a
    # shape: (n_templates_a, n_shifts_a)
    # (template_ix, shift_ix) -> shifted_template_ix
    # shifted_template_ix can be either invalid (this template does not occur
    # at this shift), or it can range from 0, ..., n_shifted_templates_a-1

    # upsampled_shifted_template_index_b
    # shape: (n_templates_b, n_shifts_b, upsampling_factor)
    # (template_ix, shift_ix, upsampling_ix) -> upsampled_shifted_template_ix

    # pconv_index
    # shape: (n_shifted_templates_a, n_upsampled_shifted_templates_b)
    # (shifted_template_ix, upsampled_shifted_template_ix) -> pconv_ix

    # pconv
    # shape: (n_pconvs, 2 * spike_length_samples - 1)
    # pconv_ix -> a cross-correlation array
    # the 0 index is special: pconv[0] === 0.

    def __init__(
        self,
        pconv: torch.Tensor,
        pconv_index: torch.Tensor,
        shifted_template_index_a: torch.Tensor,
        upsampled_shifted_template_index_b: torch.Tensor,
        shifts_a: torch.Tensor | None = None,
        shifts_b: torch.Tensor | None = None,
        on_device: bool = False,
    ):
        super().__init__()
        self.register_buffer_or_none("pconv_index", pconv_index, on_device)
        self.register_buffer_or_none("pconv", pconv, on_device)
        self.register_buffer_or_none(
            "shifted_template_index_a", shifted_template_index_a, on_device
        )
        self.register_buffer_or_none(
            "upsampled_shifted_template_index_b",
            upsampled_shifted_template_index_b,
            on_device,
        )
        self.register_buffer_or_none("shifts_a", shifts_a, on_device)
        self.register_buffer_or_none("shifts_b", shifts_b, on_device)

        # helper bufs
        na = len(self.b.shifted_template_index_a)
        self.register_buffer_or_none("all_inds_a", torch.arange(na), on_device)

        self.not_shifting = self.b.shifts_a is None
        assert self.not_shifting == (self.b.shifts_b is None)
        self.shifting = not self.not_shifting

        if self.shifting:
            shoffset_a, shindex_a = _get_shift_indexer(shifts_a)
            shoffset_a = torch.tensor(shoffset_a)
            shoffset_b, shindex_b = _get_shift_indexer(shifts_b)
            shoffset_b = torch.tensor(shoffset_b)
        else:
            shoffset_a = shindex_a = None
            shoffset_b = shindex_b = None
        self.register_buffer_or_none("shoffset_a", shoffset_a, on_device)
        self.register_buffer_or_none("shindex_a", shindex_a, on_device)
        self.register_buffer_or_none("shoffset_b", shoffset_b, on_device)
        self.register_buffer_or_none("shindex_b", shindex_b, on_device)

    @classmethod
    def from_h5(cls, hdf5_filename, on_device=False):
        with h5py.File(hdf5_filename, "r", locking=False) as h5:
            pconv = torch.from_numpy(h5["pconv"][:])  # type: ignore
            pconv_index = torch.from_numpy(h5["pconv_index"][:])  # type: ignore
            shifted_template_index_a = torch.from_numpy(
                h5["shifted_template_index_a"][:]  # type: ignore
            )
            upsampled_shifted_template_index_b = torch.from_numpy(
                h5["upsampled_shifted_template_index_b"][:]  # type: ignore
            )
            if "shifts_a" in h5:
                assert "shifts_b" in h5
                shifts_a = torch.from_numpy(h5["shifts_a"][:])  # type: ignore
                shifts_b = torch.from_numpy(h5["shifts_b"][:])  # type: ignore
            else:
                shifts_a = shifts_b = None
        return cls(
            pconv=pconv,
            pconv_index=pconv_index,
            shifted_template_index_a=shifted_template_index_a,
            upsampled_shifted_template_index_b=upsampled_shifted_template_index_b,
            shifts_a=shifts_a,
            shifts_b=shifts_b,
            on_device=on_device,
        )

    @classmethod
    def from_template_data(
        cls,
        hdf5_filename: str | Path,
        template_data: TemplateData,
        low_rank_templates: LowRankTemplates,
        compressed_upsampled_temporal: CompressedUpsampledTemplates,
        template_data_b: Optional[TemplateData] = None,
        low_rank_templates_b: Optional[LowRankTemplates] = None,
        chunk_time_centers_s: Optional[np.ndarray] = None,
        motion_est=None,
        geom: Optional[np.ndarray] = None,
        conv_batch_size=1024,
        units_batch_size=8,
        overwrite=False,
        computation_cfg=None,
        show_progress=True,
    ):
        if computation_cfg is None:
            computation_cfg = job_util.get_global_computation_config()

        hdf5_filename = resolve_path(hdf5_filename)
        hdf5_filename.parent.mkdir(exist_ok=True)

        # TODO: rewrite.
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
            conv_batch_size=conv_batch_size,
            units_batch_size=units_batch_size,
            overwrite=overwrite,
            device=computation_cfg.actual_device(),
            n_jobs=computation_cfg.actual_n_jobs(),
            show_progress=show_progress,
        )
        return cls.from_h5(hdf5_filename)

    def get_shift_ix_a(self, shifts_a):
        """Map shift (an integer, signed) to a shift index

        A shift index can be used to index into axis=1 of shifted_template_index_a,
        or self.shifts_a for that matter.
        It's an int in [0, n_shifts_a).
        It's equal to np.searchsorted(self.shifts_a, shifts_a).
        The thing is, searchsorted is slow, and we can pre-bake a lookup table.
        _get_shift_indexer does the baking for us below.
        """
        shifts_a = torch.atleast_1d(shifts_a)
        return self.b.shindex_a[shifts_a.int() + self.b.shoffset_a]

    def get_shift_ix_b(self, shifts_b):
        shifts_b = torch.atleast_1d(shifts_b)
        return self.b.shindex_b[shifts_b.int() + self.b.shoffset_b]

    def query(
        self,
        template_indices_a,
        template_indices_b,
        upsampling_indices_b=None,
        shifts_a=None,
        shifts_b=None,
        scalings_b=None,
    ):
        if template_indices_a is None:
            template_indices_a = self.b.all_inds_a
        template_indices_a = torch.atleast_1d(torch.as_tensor(template_indices_a))
        template_indices_b = torch.atleast_1d(torch.as_tensor(template_indices_b))

        # handle no shifting
        no_shifting = (shifts_a is None) or (shifts_b is None)
        shifted_template_index = self.b.shifted_template_index_a
        upsampled_shifted_template_index = self.b.upsampled_shifted_template_index_b
        if no_shifting:
            assert shifts_a is None and shifts_b is None
            assert self.b.shifts_a.shape == (1,)
            assert self.b.shifts_b.shape == (1,)
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
            assert self.b.upsampled_shifted_template_index_b.shape[2] == 1
            upsampled_shifted_template_index = upsampled_shifted_template_index[..., 0]
        else:
            b_ix = b_ix + (torch.atleast_1d(torch.as_tensor(upsampling_indices_b)),)

        # get shifted template indices for A
        shifted_temp_ix_a = shifted_template_index[a_ix]

        # upsampled shifted template indices for B
        up_shifted_temp_ix_b = upsampled_shifted_template_index[b_ix]

        pconv_indices = self.b.pconv_index[
            shifted_temp_ix_a[:, None], up_shifted_temp_ix_b[None, :]
        ].view(-1)
        (which,) = pconv_indices.nonzero(as_tuple=True)
        which_a = which // up_shifted_temp_ix_b.shape[0]
        which_b = which % up_shifted_temp_ix_b.shape[0]
        template_indices_a = template_indices_a[which_a]
        pconv_indices = pconv_indices[which]

        pconvs = self.b.pconv[pconv_indices]
        if scalings_b is not None:
            pconvs = pconvs.to(scalings_b.device)
            pconvs.mul_(scalings_b[which_b].unsqueeze(1))

        return template_indices_a, pconvs, which_b


class SeparablePairwiseConv(PconvBase):
    def __init__(self, spatial_footprints, temporal_shapes):
        """Footprint-major rank 1 template convolution database

        Let Nf = len(spatial_footprints), Ns = len(temporal_shapes). Then
        indexing is footprint-major, so that

            template[i] = spatial_footprints[i // Ns] * temporal_shapes[i - Ns * (i // Ns)]

        Let f(i) = i // Ns and s(i) = i - Ns * (i // Ns). Then the channel-summed
        convolution of templates i and j is given by

            conv(t; i, j) = (
                <spatial_footprints[f(i)], spatial_footprints[f(j)]>
                * conv1d(temporal_shapes[s(i)], temporal_shapes[s(j)])[t]
            )

        Note: need to be consistent with interpretation of the sign of the time lag
        between here and CompressedPairwiseConv.
        """
        super().__init__()
        self.register_buffer("spatial_footprints", torch.asarray(spatial_footprints))
        self.register_buffer("temporal_shapes", torch.asarray(temporal_shapes))
        self.Nf = len(spatial_footprints)
        self.Ns, self.nt = temporal_shapes.shape

        # convolve all pairs of temporal shapes
        # i is data, j is filter
        nt = temporal_shapes.shape[1]
        inp = self.b.temporal_shapes[:, None, :]
        fil = self.b.temporal_shapes[:, None, :]
        # Ns, Ns, 2*nt - 1
        self.register_buffer("tconv", F.conv1d(inp, fil, padding=nt - 1))

        # spatial component
        sdot = self.b.spatial_footprints @ self.b.spatial_footprints.T
        self.register_buffer("sdot", sdot)
        self.register_buffer("overlap", self.b.sdot > 0)
        self.tia = torch.arange(self.Ns * self.Nf)

    @property
    def device(self):
        return self.tconv.device

    def query(
        self,
        template_indices_a,
        template_indices_b,
        upsampling_indices_b=None,
        shifts_a=None,
        shifts_b=None,
        scalings_b=None,
    ):
        assert shifts_a is shifts_b is None
        assert upsampling_indices_b is None or (upsampling_indices_b == 0).all()
        if template_indices_a is None:
            template_indices_a = self.tia.to(template_indices_b)

        f_i = template_indices_a // self.Ns
        f_j = template_indices_b // self.Ns
        keep_i, keep_j = self.b.overlap[f_i[:, None], f_j[None, :]].nonzero(
            as_tuple=True
        )

        template_indices_a = template_indices_a[keep_i]
        template_indices_b = template_indices_b[keep_j]

        f_i = template_indices_a // self.Ns
        f_j = template_indices_b // self.Ns
        s_i = template_indices_a - self.Ns * f_i
        s_j = template_indices_b - self.Ns * f_j

        sdot = self.b.sdot[f_i, f_j]
        assert sdot.ndim == 1
        if scalings_b is not None:
            sdot = sdot * scalings_b[keep_j]
        tconv = self.b.tconv[s_i, s_j]
        assert tconv.shape == (len(sdot), 2 * self.nt - 1)
        pconvs = sdot.unsqueeze(1) * tconv

        return template_indices_a, pconvs, keep_j


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
