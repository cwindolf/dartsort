"""Helper functions for reassignment
"""
import numpy as np
from .deconv_resid_merge import calc_resid_matrix
from .waveform_utils import apply_tpca


def propose_pairs(
    templates,
    max_resid_dist=5,
    lambd=0.001,
    allowed_scale=0.1,
    deconv_threshold_mul=0.9,
    n_jobs=-1,
):
    """Propose overlapping template pairs using residual distance

    Returns:
        pairs : list of length len(templates)
            Contains arrays of similar template indices, such that pairs[i] always
            includes i.
    """
    n_templates = templates.shape[0]

    # shifts[i, j] is like trough[j] - trough[i]
    deconv_threshold = deconv_threshold_mul * np.min(
        np.square(templates).sum(axis=(1, 2))
    )
    # TODO: This is a sparse matrix - no need to save a n_templates * n_templates matrix
    # Breaks things when high n channels / high n templates
    resids, shifts = calc_resid_matrix(
        templates,
        np.arange(n_templates),
        templates,
        np.arange(n_templates),
        thresh=deconv_threshold,
        n_jobs=n_jobs,
        vis_ptp_thresh=1,
        auto=True,
        pbar=True,
        lambd=lambd,
        allowed_scale=allowed_scale,
    )
    np.fill_diagonal(resids, 0.0)

    # which pairs of superres templates are close enough?
    # list of superres template indices of length n_templates
    pairs = [
        np.flatnonzero(resids[i] <= max_resid_dist) for i in range(n_templates)
    ]

    return pairs


def reassign_waveforms(
    labels, cleaned_waveforms, proposed_pairs, templates_loc, tpca=None, norm_p=np.inf,
):
    N = labels.size
    assert N == cleaned_waveforms.shape[0]

    new_labels = labels.copy()
    outlier_scores = np.empty(N, dtype=cleaned_waveforms.dtype)
    for j in range(N):
        label = labels[j]
        cwf = cleaned_waveforms[j]
        pairs = proposed_pairs[label]

        resids = cwf[None, :, :] - templates_loc[label]
        resids = apply_tpca(resids, tpca)
        if norm_p == np.inf:
            scores = np.nanmax(np.abs(resids), axis=(1, 2))
        else:
            scores = np.nanmean(np.abs(resids) ** norm_p, axis=(1, 2))
        best = scores.argmin()
        new_labels[j] = pairs[best]
        outlier_scores[j] = scores[best]

    return new_labels, outlier_scores


def reassignment_templates_local(templates, proposed_pairs, channel_index):
    """
    For each template, and for each proposed pair template, get the pair
    template on the original template's channel neighborhood.
    """
    template_maxchans = templates.ptp(1).argmax(1)
    templates_padded = np.pad(
        templates,
        [(0, 0), (0, 0), (0, 1)],
        constant_values=np.nan,
    )

    reassignment_templates_loc = []
    for maxchan, pairs in zip(
        template_maxchans, proposed_pairs
    ):
        if not pairs.size:
            reassignment_templates_loc.append(np.empty([]))
            continue

        temp_chans = channel_index[maxchan]
        pair_temps_loc = templates_padded[pairs][:, :, temp_chans]
        reassignment_templates_loc.append(pair_temps_loc)

    return reassignment_templates_loc
