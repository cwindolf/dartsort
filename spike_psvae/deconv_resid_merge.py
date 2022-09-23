import numpy as np
from pathlib import Path
import tempfile
from tqdm.auto import tqdm, trange
from joblib import Parallel, delayed

from .extract_deconv import extract_deconv
from .deconvolve import MatchPursuitObjectiveUpsample
from .snr_templates import get_single_templates, get_templates
from .pre_deconv_merge_split import get_proposed_pairs
from .localize_index import localize_ptps_index


def resid_dist(
    target_template,
    search_templates,
    deconv_threshold,
    max_upsample=8,
    trough_offset=42,
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
    # spike_train_up[:, 0] += trough_offset

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
    trough_offset=42,
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
            trough_offset=trough_offset,
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
    trough_offset=42,
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
        trough_offset=trough_offset,
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


def run_deconv_merge(
    spike_train,
    geom,
    raw_binary_file,
    unit_max_channels,
    deconv_threshold_mul=0.9,
    merge_resid_threshold=1.5,
):
    templates_cleaned, extra = get_templates(
        spike_train,
        geom,
        raw_binary_file,
        unit_max_channels,
    )
    tpca = extra["tpca"]

    deconv_threshold = (
        deconv_threshold_mul
        * np.square(templates_cleaned).sum(axis=(1, 2)).min()
    )

    x, y, z_rel, z_abs, alpha = localize_ptps_index(
        templates_cleaned.ptp(1),
        geom,
        templates_cleaned.ptp(1).argmax(1),
        np.arange(len(geom))[None, :] * np.ones(len(geom), dtype=int)[:, None],
    )

    dist_argsort, dist_template = get_proposed_pairs(
        templates_cleaned.shape[0],
        templates_cleaned,
        np.c_[x, z_abs],
        n_temp=20,
    )

    with tempfile.TemporaryDirectory(prefix="drm") as workdir:
        max_values, units, units_matched, shifts = find_original_merges(
            workdir, templates_cleaned, dist_argsort, deconv_threshold
        )
        print(
            f"{max_values.shape=}, {units.shape=}, {units_matched.shape=}, {shifts.shape=}"
        )
        (
            templates_updated,
            times_updated,
            labels_updated,
            unit_reference,
        ) = merge_units_temp_deconv(
            units,
            units_matched,
            max_values,
            shifts,
            templates_cleaned,
            spike_train[:, 1].copy(),
            spike_train[:, 0].copy(),
            workdir,
            deconv_threshold,
            geom,
            raw_binary_file,
            tpca,
            merge_resid_threshold=merge_resid_threshold,
        )

    return times_updated, labels_updated
