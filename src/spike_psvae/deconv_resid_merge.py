import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import complete, fcluster

from .deconvolve import MatchPursuitObjectiveUpsample
from .snr_templates import get_templates


def resid_dist(
    target_template,
    search_templates,
    deconv_threshold,
    max_upsample=8,
    sampling_rate=30000,
    conv_approx_rank=5,
    n_processors=1,
    multi_processing=False,
    lambd=0.001,
    allowed_scale=0.1,
    distance_kind="rms",
):
    T, C = target_template.shape
    if search_templates.ndim == 2:
        search_templates = search_templates[None]
    N, T_, C_ = search_templates.shape
    assert T == T_ and C == C_

    # pad target so that the deconv can find arbitrary offset
    target_recording = np.pad(target_template, [(T, T), (0, 0)])

    mp_object = MatchPursuitObjectiveUpsample(
        templates=search_templates,
        deconv_dir=None,
        standardized_bin=None,
        t_start=0,
        t_end=None,
        n_sec_chunk=1,
        sampling_rate=30_000,
        max_iter=1,
        upsample=max_upsample,
        threshold=deconv_threshold,
        conv_approx_rank=conv_approx_rank,
        n_processors=n_processors,
        multi_processing=False,
        verbose=False,
        lambd=lambd,
        allowed_scale=allowed_scale,
    )

    mp_object.run_array(target_recording)
    deconv_st = mp_object.dec_spike_train
    if not deconv_st.shape[0]:
        return -1, np.inf, 0

    # get the rest of the information for computing the residual
    deconv_scalings = mp_object.dec_scalings
    (
        templates_up,
        deconv_id_sparse_temp_map,
    ) = mp_object.get_sparse_upsampled_templates(save_npy=False)
    deconv_id_sparse_temp_map = deconv_id_sparse_temp_map.astype(int)
    templates_up = templates_up.transpose(2, 0, 1)
    labels_up = deconv_id_sparse_temp_map[deconv_st[:, 1]]

    # subtract from target_recording to leave the residual behind
    rel_times = np.arange(T)
    for i in range(deconv_st.shape[0]):
        target_recording[deconv_st[i, 0] + rel_times] -= (
            deconv_scalings[i] * templates_up[labels_up[i]]
        )

    match_ix = int(deconv_st[0, 1] / max_upsample)
    if distance_kind == "rms":
        dist = np.sqrt(
            np.square(target_recording).sum() / (np.abs(target_recording) > 0).sum()
        )
    elif distance_kind == "max":
        dist = np.abs(target_recording).max()
    else:
        assert False
    shift = deconv_st[0, 0] - T

    return match_ix, dist, shift


def resid_dist_multiple(
    target_templates,
    search_templates,
    max_upsample=8,
    sampling_rate=30000,
    conv_approx_rank=5,
    n_processors=1,
    multi_processing=False,
    lambd=0.001,
    allowed_scale=0.1,
    deconv_threshold=None,
    distance_kind="rms",
):
    N, T, C = target_templates.shape
    if search_templates.ndim == 2:
        search_templates = search_templates[None]
    N_, T_, C_ = search_templates.shape
    assert T == T_ and C == C_ and N == N_

    # pad target so that the deconv can find arbitrary offset
    target_recording = np.pad(
        target_templates.reshape((N * T, C)), [(T, T), (0, 0)]
    )
    if deconv_threshold is None:
        deconv_threshold = 0.5 * (target_templates**2).sum((1, 2)).min()

    mp_object = MatchPursuitObjectiveUpsample(
        templates=search_templates,
        deconv_dir=None,
        standardized_bin=None,
        t_start=0,
        t_end=None,
        n_sec_chunk=1,
        sampling_rate=30_000,
        max_iter=N,
        upsample=max_upsample,
        threshold=deconv_threshold,
        conv_approx_rank=min(conv_approx_rank, C),
        n_processors=n_processors,
        multi_processing=False,
        verbose=False,
        lambd=lambd,
        allowed_scale=allowed_scale,
    )

    mp_object.run_array(target_recording)
    deconv_st = mp_object.dec_spike_train
    if not deconv_st.shape[0]:
        return -1, 0

    # get the rest of the information for computing the residual
    deconv_scalings = mp_object.dec_scalings
    (
        templates_up,
        deconv_id_sparse_temp_map,
    ) = mp_object.get_sparse_upsampled_templates(save_npy=False)
    deconv_id_sparse_temp_map = deconv_id_sparse_temp_map.astype(int)
    templates_up = templates_up.transpose(2, 0, 1)
    labels_up = deconv_id_sparse_temp_map[deconv_st[:, 1]]

    # subtract from target_recording to leave the residual behind
    rel_times = np.arange(T)
    for i in range(deconv_st.shape[0]):
        target_recording[deconv_st[i, 0] + rel_times] -= (
            deconv_scalings[i] * templates_up[labels_up[i]]
        )

    if distance_kind == "rms":
        dist = np.sqrt(
            np.square(target_recording).sum() / (target_recording != 0).sum()
        )
    elif distance_kind == "max":
        np.abs(target_recording).max()
    else:
        assert False
    shift = np.median(deconv_st[:, 0] - T * np.arange(1, len(deconv_st) + 1))

    return dist, shift


def calc_resid_matrix(
    templates_a,
    units_a,
    templates_b,
    units_b,
    thresh=8,
    thresh_mul=0.9,
    n_jobs=-1,
    vis_ptp_thresh=1,
    auto=False,
    pbar=True,
    max_upsample=8,
    lambd=0.001,
    allowed_scale=0.1,
    normalized=False,
    sampling_rate=30000,
    conv_approx_rank=5,
    distance_kind="rms",
):
    if auto and thresh is None:
        thresh = thresh_mul * np.min(np.square(templates_a).sum(axis=(1, 2)))
    # we will calculate resid dist for templates that overlap at all
    # according to these channel neighborhoods
    chans_a = [
        np.flatnonzero(temp.ptp(0) > vis_ptp_thresh) for temp in templates_a
    ]
    chans_b = [
        np.flatnonzero(temp.ptp(0) > vis_ptp_thresh) for temp in templates_b
    ]

    def job(i, j):
        return (
            i,
            j,
            *resid_dist(
                templates_a[i],
                templates_b[j],
                thresh,
                max_upsample=max_upsample,
                sampling_rate=sampling_rate,
                conv_approx_rank=conv_approx_rank,
                lambd=lambd,
                allowed_scale=allowed_scale,
                distance_kind=distance_kind,
            )
        )

    jobs = []
    resid_matrix = np.full(
        (units_a.size, units_b.size), np.inf, dtype=templates_a.dtype
    )
    shift_matrix = np.zeros((units_a.size, units_b.size), dtype=int)
    for i, ua in enumerate(units_a):
        for j, ub in enumerate(units_b):
            if auto and ua == ub:
                continue
            if np.intersect1d(chans_a[i], chans_b[j]).size:
                jobs.append(delayed(job)(i, j))

    if pbar:
        jobs = tqdm(jobs, desc="Resid matrix")
    for i, j, match_ix, dist, shift in Parallel(n_jobs)(jobs):
        resid_matrix[i, j] = dist
        shift_matrix[i, j] = shift

    if normalized:
        assert auto

        # get rms on active channels
        rms_a = np.array(
            [
                np.sqrt(np.square(ta).sum() / (np.abs(ta) > 0).sum())
                for ta in templates_a
            ]
        )
        rms_b = np.array(
            [
                np.sqrt(np.square(tb).sum() / (np.abs(tb) > 0).sum())
                for tb in templates_b
            ]
        )

        # normalize by a factor to make things dimensionless
        if distance_kind == "max":
            normresids = resid_matrix / np.sqrt(rms_a[:, None] * rms_b[None, :])
        elif distance_kind == "rms":
            normresids = resid_matrix / rms_a[:, None]
        else:
            assert False
        del resid_matrix

        # symmetrize resids and get corresponding best shifts
        resid_matrix = np.minimum(normresids, normresids.T)
        shift_matrix = np.where(
            normresids <= normresids.T,
            shift_matrix,
            -shift_matrix.T,
        )
        del normresids

    return resid_matrix, shift_matrix


def run_deconv_merge(
    spike_train,
    geom,
    raw_binary_file,
    unit_max_channels=None,
    deconv_threshold_mul=0.9,
    merge_resid_threshold=0.25,
    tpca=None,
    trough_offset=42,
    spike_length_samples=121,
    normalized=True,
    distance_kind="rms",
):
    templates_cleaned, extra = get_templates(
        spike_train,
        geom,
        raw_binary_file,
        unit_max_channels=unit_max_channels,
        tpca=tpca,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )

    deconv_threshold = (
        deconv_threshold_mul
        * np.square(templates_cleaned).sum(axis=(1, 2)).min()
    )

    resids, shifts = calc_resid_matrix(
        templates_cleaned,
        np.arange(templates_cleaned.shape[0]),
        templates_cleaned,
        np.arange(templates_cleaned.shape[0]),
        thresh=deconv_threshold,
        n_jobs=-1,
        vis_ptp_thresh=1,
        auto=True,
        pbar=True,
        lambd=0.001,
        allowed_scale=0.1,
        normalized=normalized,
        distance_kind=distance_kind,
    )
    # shifts[i, j] is like trough[j] - trough[i]

    # upper triangle not including diagonal, aka condensed distance matrix in scipy
    pdist = resids[np.triu_indices(resids.shape[0], k=1)]
    # scipy hierarchical clustering only supports finite values, so let's just
    # drop in a huge value here
    pdist[~np.isfinite(pdist)] = 1_000_000 + pdist[np.isfinite(pdist)].max()
    # complete linkage: max dist between all pairs across clusters.
    Z = complete(pdist)
    # extract flat clustering using our max dist threshold
    new_labels = fcluster(Z, merge_resid_threshold, criterion="distance")

    # update labels
    labels_updated = spike_train[:, 1].copy()
    kept = np.flatnonzero(labels_updated >= 0)
    labels_updated[kept] = new_labels[labels_updated[kept]]

    # update times according to shifts
    times_updated = spike_train[:, 0].copy()

    # this is done by aligning each unit to the max snr unit in its cluster
    maxsnrs = extra["snr_by_channel"].max(axis=1)

    # find original labels in each cluster
    clust_inverse = {i: [] for i in new_labels}
    for orig_label, new_label in enumerate(new_labels):
        clust_inverse[new_label].append(orig_label)

    # align to best snr unit
    for new_label, orig_labels in clust_inverse.items():
        # we don't need to realign clusters which didn't change
        if len(orig_labels) <= 1:
            continue

        orig_snrs = maxsnrs[orig_labels]
        best_orig = orig_labels[orig_snrs.argmax()]
        for ogl in np.setdiff1d(orig_labels, [best_orig]):
            in_orig_unit = np.flatnonzero(spike_train[:, 1] == ogl)
            # this is like trough[best] - trough[ogl]
            shift_og_best = shifts[ogl, best_orig]
            # if >0, trough of og is behind trough of best.
            # subtracting will move trough of og to the right.
            times_updated[in_orig_unit] -= shift_og_best

    return times_updated, labels_updated
