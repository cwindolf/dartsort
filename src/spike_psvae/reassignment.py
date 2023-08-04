"""Helper functions for reassignment
"""
import numpy as np
from scipy.spatial.distance import cdist
from .deconv_resid_merge import calc_resid_matrix
from .waveform_utils import apply_tpca
from .localize_index import localize_ptps_index


def propose_pairs(
    templates,
    method="resid_dist",
    max_resid_dist=8,
    resid_dist_kwargs=dict(
        lambd=0.001,
        allowed_scale=0.1,
        thresh_mul=0.9,
        normalized=True,
        n_jobs=-1,
    ),
    max_distance=50,
    loc_radius=100,
    geom=None,
):
    """Propose overlapping template pairs using residual distance

    Returns:
        pairs : list of length len(templates)
            Contains arrays of similar template indices, such that pairs[i] always
            includes i.
    """
    n_templates = templates.shape[0]

    if method == "resid_dist":
        # TODO: This is a sparse matrix - no need to save a n_templates * n_templates matrix
        # Breaks things when high n channels / high n templates
        resids, shifts = calc_resid_matrix(
            templates,
            np.arange(n_templates),
            templates,
            np.arange(n_templates),
            thresh=None,
            vis_ptp_thresh=1,
            auto=True,
            pbar=True,
            **resid_dist_kwargs,
        )
        np.fill_diagonal(resids, 0.0)

        # which pairs of superres templates are close enough?
        # list of superres template indices of length n_templates
        pairs = [
            np.flatnonzero(resids[i] <= max_resid_dist)
            for i in range(n_templates)
        ]
    elif method == "radius":
        tx, ty, _, tz, ta = localize_ptps_index(
            templates.ptp(1),
            geom,
            templates.ptp(1).argmax(1),
            np.array([np.arange(geom.shape[0])] * geom.shape[0]),
            radius=loc_radius,
        )
        txz = np.c_[tx, tz]
        dist = cdist(txz, txz)

        pairs = [
            np.flatnonzero(dist[i] <= max_distance) for i in range(n_templates)
        ]
    else:
        assert False

    assert len(pairs) == n_templates

    return pairs


def reassign_waveforms(
    labels,
    cleaned_waveforms,
    proposed_pairs,
    templates_loc,
    tpca=None,
    norm_p=np.inf,
    return_resids=False,
):
    N = labels.size
    assert N == cleaned_waveforms.shape[0]

    new_labels = labels.copy()
    outlier_scores = np.empty(N, dtype=cleaned_waveforms.dtype)
    if return_resids:
        orig_and_reas_scores = np.empty((N, 2), dtype=cleaned_waveforms.dtype)
        orig_cleaned_wfs = cleaned_waveforms.copy()
        orig_and_reas_resids = np.empty(
            (N, 2, *cleaned_waveforms.shape[1:]), dtype=cleaned_waveforms.dtype
        )

    for j in range(N):
        label = labels[j]
        cwf = cleaned_waveforms[j]
        pairs = proposed_pairs[label]

        if pairs.size:
            resids = cwf[None, :, :] - templates_loc[label]
            resids = apply_tpca(resids, tpca)
            if norm_p == np.inf:
                scores = np.nanmax(np.abs(resids), axis=(1, 2))
            else:
                scores = np.nanmean(np.abs(resids) ** norm_p, axis=(1, 2))
            best = scores.argmin()
            new_labels[j] = pairs[best]
            outlier_scores[j] = scores[best]

        if return_resids:
            orig_label_pos = np.flatnonzero(pairs == label)
            # print(f"{np.array2string(scores, precision=4, suppress_small=True)=}")
            assert orig_label_pos.size == 1
            # print(f"{label=} {pairs=}")
            # print(f"{label=} {best=} {pairs[best]=}")
            # print(f"{label=} {orig_label_pos=} {pairs[orig_label_pos]=}")
            orig_and_reas_scores[j, 0] = scores[orig_label_pos]
            orig_and_reas_scores[j, 1] = scores[best]
            # print(f"{label=} {best=} {scores[orig_label_pos]=} {scores[best]=}")
            orig_and_reas_resids[j, 0] = resids[orig_label_pos]
            orig_and_reas_resids[j, 1] = resids[best]

            # print("---")
            # print(f"{label=} {pairs.size=} {np.flatnonzero(pairs == label)=} {cwf.shape=}")
            # print(f"{best=} {orig_label_pos=} {pairs[best]=}")
            # print(f"{(best == orig_label_pos)=} {(pairs[best] == label)=}")
            # print(f"{scores[orig_label_pos]=} {scores[best]=}")
            # print(f"{scores.size=} {np.unique(scores).size=}")
            # print(f"{np.isfinite(scores).all()=}")
            # rf = np.nan_to_num(resids)
            # req = (rf[None, :] == rf[:, None]).all(axis=(2, 3))
            # print(f"{(req.sum() - pairs.size)=}")
            # import matplotlib.pyplot as plt
            # from .cluster_viz_index import pgeom
            # import colorcet as cc
            # fig, ax = plt.subplots(figsize=(6, 6))
            # ci = np.arange(cwf.shape[1])[None, :] * np.ones(cwf.shape[1], dtype=int)[:, None]
            # g = np.c_[np.zeros(cwf.shape[1]), np.arange(cwf.shape[1])]
            # pgeom(cwf[None], max_channels=[0], channel_index=ci, geom=g, ax=ax, color="k", max_abs_amp=np.abs(cwf).max())
            # for j, tt in enumerate(templates_loc[label]):
            #     pgeom(tt[None], max_channels=[0], channel_index=ci, geom=g, ax=ax, color=cc.glasbey[j], max_abs_amp=np.abs(cwf).max())
            # plt.show()
            # plt.close(fig)

            # fig, ax = plt.subplots(figsize=(6, 6))
            # ci = np.arange(cwf.shape[1])[None, :] * np.ones(cwf.shape[1], dtype=int)[:, None]
            # g = np.c_[np.zeros(cwf.shape[1]), np.arange(cwf.shape[1])]
            # # pgeom(cwf[None], max_channels=[0], channel_index=ci, geom=g, ax=ax, color="k", max_abs_amp=np.abs(cwf).max())
            # for j, tt in enumerate(resids):
            #     pgeom(tt[None], max_channels=[0], channel_index=ci, geom=g, ax=ax, color=cc.glasbey[j], max_abs_amp=np.abs(cwf).max())
            # plt.show()
            # plt.close(fig)

    if return_resids:
        return (
            new_labels,
            outlier_scores,
            orig_cleaned_wfs,
            orig_and_reas_scores,
            orig_and_reas_resids,
        )

    return new_labels, outlier_scores


def reassignment_templates_local(templates, proposed_pairs, channel_index):
    """
    For each template, and for each proposed pair template, get the pair
    template on the original template's channel neighborhood.
    """
    assert len(proposed_pairs) == len(templates)
    assert channel_index.shape[0] == templates.shape[2]
    print(f"loc temps {len(proposed_pairs)=} {templates.shape=}")
    tmc = templates.ptp(1).argmax(1)
    print(f"{np.abs(templates)[np.arange(len(templates)), :, tmc].argmax(1)=}")

    template_maxchans = templates.ptp(1).argmax(1)
    templates_padded = np.pad(
        templates,
        [(0, 0), (0, 0), (0, 1)],
        constant_values=np.nan,
    )
    # print(f"{templates_padded.shape=}")

    reassignment_templates_loc = []
    for maxchan, pairs in zip(template_maxchans, proposed_pairs):
        if not pairs.size:
            reassignment_templates_loc.append(np.empty([0]))
            continue

        temp_chans = channel_index[maxchan]
        # print(f"{pairs.size=} {pairs.shape=} {temp_chans.shape=} {pairs=}")
        pair_temps_loc = templates_padded[pairs][:, :, temp_chans]
        # print(f"{pair_temps_loc.shape=}")
        # ptmc = np.nanargmax(pair_temps_loc.ptp(1), 1)
        # print(f"{ptmc=}")
        # print(f"{np.abs(pair_temps_loc)[np.arange(len(pair_temps_loc)), :, ptmc].argmax(1)=}")
        reassignment_templates_loc.append(pair_temps_loc)

    return reassignment_templates_loc
