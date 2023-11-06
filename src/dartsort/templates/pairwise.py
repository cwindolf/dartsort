from dataclasses import dataclass, fields

import h5py
import numpy as np


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
            upsampled_shifted_template_index = self.upsampled_shifted_template_index[..., 0]
        else:
            b_ix = b_ix + (upsampling_indices_b,)

        # get shifted template indices for A
        shifted_temp_ix_a = shifted_template_index[a_ix]

        # upsampled shifted template indices for B
        up_shifted_temp_ix_b = upsampled_shifted_template_index[b_ix]

        pconv_indices = self.pconv_index[shifted_temp_ix_a, up_shifted_temp_ix_b]

        # most users will be happy not to get a bunch of zeros for pairs that don't overlap
        if not return_zero_convs:
            which = np.flatnonzero(pconv_indices > 0)
            pconv_indices = pconv_indices[which]
            template_indices_a = template_indices_a[which]
            template_indices_b = template_indices_b[which]

        return template_indices_a, template_indices_b, self.pconv[pconv_indices]



@dataclass
class SparsePairwiseConv:
    # shift_ix -> shift
    shifts: np.ndarray
    # (temp_ix, shift_ix) -> shifted_temp_ix
    template_shift_index: torch.LongTensor
    # (shifted_temp_ix a, shifted_temp_ix b) -> pair index
    pair_index_table: torch.LongTensor
    # (pair index, upsampling index) -> pconv index
    upsampling_index_table: torch.LongTensor
    # pconv index -> pconv (2 * spike len - 1,)
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
        template_indices_a = np.atleast_1d(template_indices_a)
        template_indices_b = np.atleast_1d(template_indices_b)
        shifted = shifts_a is not None
        if shifted:
            assert shifts_b is not None
            shifts_a = np.atleast_1d(shifts_a)
            shifts_b = np.atleast_1d(shifts_b)
        else:
            assert np.array_equal(self.shifts, [0.0])

        # get shifted template indices
        if shifted:
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
        else:
            shifted_temp_ix_a = template_indices_a
            shifted_temp_ix_b = template_indices_b

        # we only store the upper triangle of this symmetric object
        min_ = np.minimum(shifted_temp_ix_a, shifted_temp_ix_b)
        max_ = np.maximum(shifted_temp_ix_a, shifted_temp_ix_b)
        pair_indices = self.pair_index_table[min_, max_]

        # handle upsampling
        if upsampling_indices_b is None:
            assert self.upsampling_index_table.shape[1] == 1
            pconv_indices = self.upsampling_index_table[pair_indices, 0]
        else:
            pconv_indices = self.upsampling_index_table[
                pair_indices, upsampling_indices_b
            ]

        # most users will be happy not to get a bunch of zeros for pairs that don't overlap
        if not return_zero_convs:
            which = np.flatnonzero(pconv_indices > 0)
            pconv_indices = pconv_indices[which]
            template_indices_a = template_indices_a[which]
            template_indices_b = template_indices_b[which]

        pair_convs = self.pconv[pconv_indices]

        return template_indices_a, template_indices_b, pair_convs
