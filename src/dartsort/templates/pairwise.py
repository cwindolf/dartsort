from dataclasses import dataclass, fields
from typing import Optional

import h5py
import numpy as np
import torch

from .pairwise_util import compressed_convolve_to_h5
from .template_util import CompressedUpsampledTemplates, LowRankTemplates
from .templates import TemplateData


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
    shifts: np.ndarray

    # shape: (n_templates, n_shifts)
    # (template_ix, shift_ix) -> shifted_template_ix
    # shifted_template_ix can be either invalid (this template does not occur
    # at this shift), or it can range from 0, ..., n_shifted_templates-1
    shifted_template_index: np.ndarray

    # shape: (n_templates, n_shifts, upsampling_factor)
    # (template_ix, shift_ix, upsampling_ix) -> upsampled_shifted_template_ix
    upsampled_shifted_template_index: np.ndarray

    # shape: (n_shifted_templates, n_upsampled_shifted_templates)
    # (shifted_template_ix, upsampled_shifted_template_ix) -> pconv_ix
    pconv_index: np.ndarray

    # shape: (n_pconvs, 2 * spike_length_samples - 1)
    # pconv_ix -> a cross-correlation array
    # the 0 index is special: pconv[0] === 0.
    pconv: np.ndarray

    def __post_init__(self):
        assert self.shifts.ndim == 1
        assert self.shifts.size == self.shifted_template_index.shape[1]
        assert (
            self.shifted_template_index.shape
            == self.upsampled_shifted_template_index.shape[:2]
        )
        self._is_torch = False

    @classmethod
    def from_h5(cls, hdf5_filename):
        ff = fields(cls)
        with h5py.File(hdf5_filename, "r") as h5:
            data = {f.name: h5[f.name][:] for f in ff}
        return cls(**data)

    @classmethod
    def from_template_data(
        cls,
        hdf5_filename,
        template_data: TemplateData,
        low_rank_templates: LowRankTemplates,
        compressed_upsampled_temporal: CompressedUpsampledTemplates,
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
        print(f"pairwise from_template_data {device=}")
        compressed_convolve_to_h5(
            hdf5_filename,
            template_data=template_data,
            low_rank_templates=low_rank_templates,
            compressed_upsampled_temporal=compressed_upsampled_temporal,
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

    def at_shifts(self, shifts=None):
        """Subset this database to one set of shifts.

        The database becomes shiftless (not in the pejorative sense).
        """
        if shifts is None:
            assert self.shifts.shape == (1,)
            return self

        assert shifts.shape == len(self.shifted_template_index)
        n_shifted_temps, n_up_shifted_temps = self.pconv_index.shape

        # active shifted and upsampled indices
        shift_ix = np.searchsorted(self.shifts, shifts)
        sub_shifted_temp_index = self.shifted_template_index[
            np.arange(len(self.shifted_template_index)),
            shift_ix,
        ]
        sub_up_shifted_temp_index = self.upsampled_shifted_template_index[
            np.arange(len(self.shifted_template_index)),
            shift_ix,
        ]

        # in flat form for indexing into pconv_index. also, reindex.
        valid_shifted = sub_shifted_temp_index < n_shifted_temps
        shifted_temp_ixs, new_shifted_temp_ixs = np.unique(
            sub_shifted_temp_index[valid_shifted]
        )
        valid_up_shifted = sub_up_shifted_temp_index < n_up_shifted_temps
        up_shifted_temp_ixs, new_up_shifted_temp_ixs = np.unique(
            sub_up_shifted_temp_index[valid_up_shifted], return_inverse=True
        )

        # get relevant pconv subset and reindex
        sub_pconv_indices, new_pconv_indices = np.unique(
            self.pconv_index[
                shifted_temp_ixs[:, None],
                up_shifted_temp_ixs.ravel()[None, :],
            ],
            return_inverse=True,
        )
        sub_pconv = self.pconv[sub_pconv_indices]

        # reindexing
        n_sub_shifted_temps = len(shifted_temp_ixs)
        n_sub_up_shifted_temps = len(up_shifted_temp_ixs)
        sub_pconv_index = new_pconv_indices.reshape(
            n_sub_shifted_temps, n_sub_up_shifted_temps
        )
        sub_shifted_temp_index[valid_shifted] = new_shifted_temp_ixs
        sub_up_shifted_temp_index[valid_shifted] = new_up_shifted_temp_ixs

        return self.__class__(
            shifts=np.zeros(1),
            shifted_template_index=sub_shifted_temp_index,
            upsampled_shifted_template_index=sub_up_shifted_temp_index,
            pconv_index=sub_pconv_index,
            pconv=sub_pconv,
        )

    def to(self, device=None):
        """Become torch tensors on device."""
        for f in fields(self):
            self.setattr(f.name, torch.as_tensor(getattr(self, f.name), device=device))
        self.device = device

    def query(
        self,
        template_indices_a,
        template_indices_b,
        upsampling_indices_b=None,
        shifts_a=None,
        shifts_b=None,
        return_zero_convs=False,
        grid=False,
    ):
        if template_indices_a is None:
            if self._is_torch:
                template_indices_a = torch.arange(
                    len(self.shifted_template_index), device=self.device
                )
            else:
                template_indices_a = np.arange(len(self.shifted_template_index))
        if not self._is_torch:
            template_indices_a = np.atleast_1d(template_indices_a)
            template_indices_b = np.atleast_1d(template_indices_b)

        # handle no shifting
        no_shifting = shifts_a is None or shifts_b is None
        shifted_template_index = self.shifted_template_index
        upsampled_shifted_template_index = self.upsampled_shifted_template_index
        if no_shifting:
            assert shifts_a is None and shifts_b is None
            assert self.shifts.shape == (1,)
            a_ix = (template_indices_a,)
            b_ix = (template_indices_b,)
            shifted_template_index = shifted_template_index[:, 0]
            upsampled_shifted_template_index = upsampled_shifted_template_index[:, 0]
        else:
            shift_indices_a = np.searchsorted(self.shifts, shifts_a)
            shift_indices_b = np.searchsorted(self.shifts, shifts_b)
            a_ix = (template_indices_a, shift_indices_a)
            b_ix = (template_indices_b, shift_indices_b)

        # handle no upsampling
        no_upsampling = upsampling_indices_b is None
        if no_upsampling:
            assert self.upsampled_shifted_template_index.shape[2] == 1
            upsampled_shifted_template_index = upsampled_shifted_template_index[..., 0]
        else:
            b_ix = b_ix + (upsampling_indices_b,)

        # get shifted template indices for A
        shifted_temp_ix_a = shifted_template_index[a_ix]

        # upsampled shifted template indices for B
        up_shifted_temp_ix_b = upsampled_shifted_template_index[b_ix]

        # return convolutions between all ai,bj or just ai,bi?
        if grid:
            pconv_indices = self.pconv_index[shifted_temp_ix_a[:, None], up_shifted_temp_ix_b[None, :]]
            if self._is_torch:
                template_indices_a, template_indices_b = torch.cartesian_prod(
                    template_indices_a, template_indices_b
                ).T
                pconv_indices = pconv_indices.view(-1)
            else:
                template_indices_a, template_indices_b = np.meshgrid(template_indices_a, template_indices_b, indexing="ij")
                template_indices_a = template_indices_a.ravel()
                template_indices_b = template_indices_b.ravel()
                pconv_indices = pconv_indices.ravel()
        else:
            pconv_indices = self.pconv_index[shifted_temp_ix_a, up_shifted_temp_ix_b]

        # most users will be happy not to get a bunch of zeros for pairs that don't overlap
        if not return_zero_convs:
            which = pconv_indices > 0
            pconv_indices = pconv_indices[which]
            template_indices_a = template_indices_a[which]
            template_indices_b = template_indices_b[which]

        return template_indices_a, template_indices_b, self.pconv[pconv_indices]
