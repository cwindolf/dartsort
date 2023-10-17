from dataclasses import dataclass


def sparse_pairwise_conv(
    sorting,
    template_temporal_components,
    template_upsampled_temporal_components,
    template_singular_values,
    template_spatial_components,
    conv_ignore_threshold: float = 0.0,
    coarse_approx_error_threshold: float = 0.0,
):
    """
    
    Arguments
    ---------
    sorting : DARTsortSorting
        original (non-superres) sorting. its labels should appear in
        template_data.unit_ids
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
    
    
    

def _pairwise_conv_job(
    units_a,
    units_b,
):
    """units_a,b are chunks of original (non-superres) unit labels"""
    # determine co-visibility
    # get all coarse templates
    # get all superres templates
    # compute all coarse and superres pconvs
    
    # returns
    # list of tuples containing:
    #  - pitch shift ix a
    #  - pitch shift ix b
    #  - superres label a
    #  - superres label b
    # list of the same length containing:
    #  - -1 or an index into the next list
    # list of pconvs, indexed by previous list
    