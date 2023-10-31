from dataclasses import dataclass


def sparse_pairwise_conv(
    template_data,
    template_temporal_components,
    template_upsampled_temporal_components,
    template_singular_values,
    template_spatial_components,
    chunk_time_centers_s=None,
    motion_est=None,
    conv_ignore_threshold: float = 0.0,
    coarse_approx_error_threshold: float = 0.0,
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
            -1
            if superres pconv a,b at these shifts was below the conv_ignore_threshold
            else pconv_index)
    pconvs: np.ndarray
        pconv[pconv_index] is a cross-correlation of two templates, summed over chans
    """
    # find all of the co-occurring pitch shift and unit id pairs
    all_pitch_shifts, shift_unit_pairs = get_shift_and_unit_pairs(
        chunk_time_centers_s,
        geom,
        template_data,
        motion_est=motion_est,
    )


# defining this dtype, which represents a pair of units and shifts,
# allows us to use numpy's 1d set functions on these pairs
shift_unit_pair_dtype = np.dtype(
    [("unita", int), ("shifta", int), ("unitb", int), ("shiftb", int)]
)


class PairwiseConvContext:
    def __init__(
        self,
        coarse_spatial,
        coarse_singular,
        coarse_f_temporal,
        spatial,
        singular,
        f_temporal,
        f_temporal_up,
        geom,
        registered_geom,
    ):


def _pairwise_conv_job(
    units_a,
    units_b,
):
    """units_a,b are chunks of original (non-superres) unit labels"""
    
    # returns
    # array of type shift_unit_pair_dtype
    # array of the same length containing
    #  - -1 or an index into the next list
    # list of pconvs, indexed by previous list

    # determine co-visible shift/unit pairs
    
    # extract template data for left and right entries of each
    # pair into npairs-len structures
    # "depthwise" convolve these two structures

    # same for the coarse templates
    
    # when max pconv is < co-correlation threshold:
    #  - key list entry gets -1
    
    # now, the coarse part
    # for each pair of coarse units, check if the max difference
    # of coarse pconv and all superres pconvs is small enough,
    # and use an id for the (temporally upsampled) coarse pconv if so
    

    pass



def get_shift_and_unit_pairs(
    chunk_time_centers_s,
    geom,
    template_data,
    motion_est=None,
):
    if motion_est is None:
        return None, None

    # all observed pitch shift values
    all_pitch_shifts = []
    # set of (unit a, shift a, unit b, shift b)
    # units are unit ids, not (superres) template indices
    shift_unit_pairs = []

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

        # get unique pitch/unit shift pairs in chunk
        pitch_and_unit = np.c_[td.unit_ids, pitch_shifts.astype(int)]
        pairs = np.concatenate(
            np.broadcast_arrays(
                pitch_and_unit[:, None, :],
                pitch_and_unit[None, :, :],
            ),
            axis=2,
        )
        pairs = pairs.reshape(len(td.unit_ids) ** 2, 4)
        pairs = np.ascontiguousarray(pairs).view(shift_unit_pair_dtype)
        unique_pairs_in_chunk = np.unique(pairs)

        # update full set
        all_pitch_shifts = np.union1d(all_pitch_shifts, pitch_shifts)
        shift_unit_pairs = np.union1d(shift_unit_pairs, unique_pairs_in_chunk)
    
    return all_pitch_shifts, shift_unit_pairs
