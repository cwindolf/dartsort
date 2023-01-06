import numpy as np
from tqdm.auto import tqdm, trange
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import complete, fcluster

from .deconvolve import MatchPursuitObjectiveUpsample
from .snr_templates import get_single_templates, get_templates


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
    dist = np.abs(target_recording).max()
    shift = deconv_st[0, 0] - T

    return match_ix, dist, shift


def find_original_merges(
    templates_cleaned,
    dist_argsort,
    deconv_threshold,
    max_upsample=8,
    n_pairs_proposed=10,
    sampling_rate=30000,
    conv_approx_rank=5,
    lambd=0.001,
    allowed_scale=0.01,
    n_proposals=20,
    n_jobs=-1,
):
    N, T, C = templates_cleaned.shape

    max_values = []
    units = []
    units_matched = []
    shifts = []

    def job(i):
        templates_cleaned_amputated = templates_cleaned[
            dist_argsort[i][:n_pairs_proposed]
        ]
        match_ix, dist, shift = resid_dist(
            templates_cleaned[i],
            templates_cleaned_amputated,
            deconv_threshold,
            max_upsample=max_upsample,
            sampling_rate=sampling_rate,
            conv_approx_rank=conv_approx_rank,
            n_processors=1,
            multi_processing=False,
            lambd=lambd,
            allowed_scale=allowed_scale,
        )
        return i, match_ix, dist, shift

    with Parallel(n_jobs) as p:
        for i, match_ix, dist, shift in p(
            delayed(job)(i)
            for i in trange(templates_cleaned.shape[0], desc="Original merges")
        ):
            max_values.append(dist)
            units.append(i)
            units_matched.append(dist_argsort[i][match_ix])
            shifts.append(shift)

    return (
        np.array(max_values),
        np.array(units),
        np.array(units_matched),
        np.array(shifts),
    )


def check_additional_merge(
    temp_to_input,
    temp_to_deconv,
    deconv_threshold,
    max_upsample=8,
    n_pairs_proposed=10,
    sampling_rate=30000,
    conv_approx_rank=5,
    lambd=0.001,
    allowed_scale=0.1,
):
    match_ix, dist, shift = resid_dist(
        temp_to_deconv,
        temp_to_input,
        deconv_threshold,
        max_upsample=max_upsample,
        sampling_rate=sampling_rate,
        conv_approx_rank=conv_approx_rank,
        lambd=lambd,
        allowed_scale=allowed_scale,
    )

    return dist, shift


def merge_units_temp_deconv(
    units,
    units_matched,
    max_values,
    shifts,
    templates_cleaned,
    labels,
    spike_times,
    deconv_threshold,
    geom,
    raw_bin,
    tpca,
    merge_resid_threshold=1.5,
):
    templates_updated = templates_cleaned.copy()
    labels_updated = labels.copy()
    spike_times = spike_times.copy()

    units_already_merged = []
    unit_reference = np.arange(templates_cleaned.shape[0])

    units = units[max_values <= merge_resid_threshold]
    units_matched = units_matched[max_values <= merge_resid_threshold]
    shifts = shifts[max_values <= merge_resid_threshold]
    max_values = max_values[max_values <= merge_resid_threshold]

    idx = max_values.argsort()

    units = units[idx]
    units_matched = units_matched[idx]
    max_values = max_values[idx]
    shifts = shifts[idx]

    for j, unit, matched in tqdm(
        zip(range(len(units)), units, units_matched),
        desc="Deconv merge",
        total=len(units),
    ):
        if ~np.isin(unit, units_already_merged) and ~np.isin(
            matched, units_already_merged
        ):
            # MERGE unit, units_matched[j]
            units_already_merged.append(unit)
            units_already_merged.append(matched)
            unit_reference[matched] = unit
            labels_updated[labels_updated == matched] = unit

            # Update spike times
            spike_times[labels == matched] -= shifts[j]
            # Update template
            spike_times_test = spike_times[np.isin(labels, [matched, unit])]
            temp_merge = get_single_templates(
                spike_times_test, geom, raw_bin, tpca
            )
            templates_updated[matched] = temp_merge
            templates_updated[unit] = temp_merge

        elif np.isin(unit, units_already_merged) and ~np.isin(
            matched, units_already_merged
        ):
            # check MERGE matched to unit
            unit_ref = unit_reference[unit]
            temp_to_input = templates_cleaned[matched]
            temp_to_deconv = templates_updated[unit_ref]
            maxresid, shift = check_additional_merge(
                temp_to_input, temp_to_deconv, deconv_threshold
            )

            if maxresid < merge_resid_threshold:
                units_already_merged.append(matched)
                unit_reference[matched] = unit_ref

                # Update spike times
                spike_times[labels_updated == matched] -= shift
                spike_times_test = spike_times[
                    np.isin(labels, [matched, unit, unit_ref])
                ]
                temp_merge = get_single_templates(
                    spike_times_test, geom, raw_bin, tpca
                )
                templates_updated[matched] = temp_merge
                templates_updated[unit] = temp_merge
                templates_updated[unit_ref] = temp_merge
                labels_updated[labels_updated == matched] = unit_ref

        elif ~np.isin(unit, units_already_merged) and np.isin(
            matched, units_already_merged
        ):
            # check MERGE unit to matched
            unit_ref = unit_reference[matched]
            temp_to_input = templates_cleaned[unit_ref]
            temp_to_deconv = templates_updated[unit]

            maxresid, shift = check_additional_merge(
                temp_to_input, temp_to_deconv, deconv_threshold
            )

            if maxresid < merge_resid_threshold:
                units_already_merged.append(unit)
                unit_reference[unit] = unit_ref

                # Update spike times
                # spike_times[labels_updated == unit] += shifts[j]
                spike_times[labels_updated == unit] += shift
                spike_times_test = spike_times[
                    np.isin(labels, [matched, unit, unit_ref])
                ]
                temp_merge = get_single_templates(
                    spike_times_test, geom, raw_bin, tpca
                )
                templates_updated[matched] = temp_merge
                templates_updated[unit] = temp_merge
                templates_updated[unit_ref] = temp_merge
                labels_updated[labels_updated == unit] = unit_ref

        else:
            # check MERGE unit_reference[matched] to unit_reference[unit]
            temp_to_input = templates_cleaned[unit_reference[matched]]
            temp_to_deconv = templates_updated[unit_reference[unit]]

            maxresid, shift = check_additional_merge(
                temp_to_input, temp_to_deconv, deconv_threshold
            )

            if maxresid < merge_resid_threshold:
                unit_reference[matched] = unit_reference[unit]
                unit_reference[unit_reference[matched]] = unit_reference[unit]

                # Update spike times
                spike_times[labels_updated == unit_reference[matched]] -= shift
                # ] -= shifts[j]
                spike_times_test = spike_times[
                    np.isin(
                        labels,
                        [
                            matched,
                            unit,
                            unit_reference[unit],
                            unit_reference[matched],
                        ],
                    ),
                ]
                temp_merge = get_single_templates(
                    spike_times_test, geom, raw_bin, tpca
                )
                templates_updated[matched] = temp_merge
                templates_updated[unit] = temp_merge
                templates_updated[unit_reference[unit]] = temp_merge
                templates_updated[unit_reference[matched]] = temp_merge
                labels_updated[labels_updated == matched] = unit_reference[
                    unit
                ]
                labels_updated[
                    labels_updated == unit_reference[matched]
                ] = unit_reference[unit]

    return templates_updated, spike_times, labels_updated, unit_reference


def resid_dist__(temp_a, temp_b, thresh, lambd=0.001, allowed_scale=0.1):
    maxres_a, shift_a = check_additional_merge(
        temp_a, temp_b, thresh, lambd=lambd, allowed_scale=allowed_scale
    )
    # maxres_b, shift_b = deconv_resid_merge.check_additional_merge(
    #     temp_b, temp_a, thresh, lambd=lambd, allowed_scale=allowed_scale
    # )
    # shift from a -> b
    return maxres_a, shift_a


def calc_resid_matrix(
    templates_a,
    units_a,
    templates_b,
    units_b,
    thresh=8,
    n_jobs=-1,
    vis_ptp_thresh=1,
    auto=False,
    pbar=True,
    lambd=0.001,
    allowed_scale=0.1,
):
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
            *resid_dist__(
                templates_a[i],
                templates_b[j],
                thresh,
                lambd=lambd,
                allowed_scale=allowed_scale,
            ),
        )

    jobs = []
    resid_matrix = np.full((units_a.size, units_b.size), np.inf)
    shift_matrix = np.zeros((units_a.size, units_b.size))
    for i, ua in enumerate(units_a):
        for j, ub in enumerate(units_b):
            if auto and ua == ub:
                continue
            if np.intersect1d(chans_a[i], chans_b[j]).size:
                jobs.append(delayed(job)(i, j))

    if pbar:
        jobs = tqdm(jobs, desc="Resid matrix")
    for i, j, dist, shift in Parallel(n_jobs)(jobs):
        resid_matrix[i, j] = dist
        shift_matrix[i, j] = shift

    return resid_matrix, shift_matrix


def run_deconv_merge(
    spike_train,
    geom,
    raw_binary_file,
    unit_max_channels=None,
    deconv_threshold_mul=0.9,
    # 2 is conservative, 2.5 is nice, 3 is aggressive
    merge_resid_threshold=2.5,
    tpca=None,
    trough_offset=42,
    spike_length_samples=121,
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

    # get rms on active channels
    rms = np.array(
        [
            np.sqrt(np.square(ta).sum() / (np.abs(ta) > 0).sum())
            for ta in templates_cleaned
        ]
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
    )
    # shifts[i, j] is like trough[j] - trough[i]

    # normalize by a factor to make things dimensionless
    normresids = resids / np.sqrt(rms[:, None] * rms[None, :])
    del resids

    # symmetrize resids and get corresponding best shifts
    symresids = np.minimum(normresids, normresids.T)
    symshifts = np.where(
        normresids <= normresids.T,
        shifts,
        -shifts.T,
    ).astype(int)
    del normresids

    # upper triangle not including diagonal, aka condensed distance matrix in scipy
    pdist = symresids[np.triu_indices(symresids.shape[0], k=1)]
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
            shift_og_best = symshifts[ogl, best_orig]
            # if >0, trough of og is behind trough of best.
            # subtracting will move trough of og to the right.
            times_updated[in_orig_unit] -= shift_og_best

    return times_updated, labels_updated
