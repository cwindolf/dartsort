# %%
import numpy as np
from pathlib import Path
import tempfile
from tqdm.auto import tqdm

# %%
from .extract_deconv import extract_deconv
from .deconvolve import MatchPursuitObjectiveUpsample
from .snr_templates import get_single_templates, get_templates
from .pre_deconv_merge_split import get_proposed_pairs
from .localize_index import localize_ptps_index


# %%
def find_original_merges(
    output_dir,
    templates_cleaned,
    dist_argsort,
    deconv_threshold,
    max_upsample=8,
    trough_offset=42,
    n_pairs_proposed=10,
    sampling_rate=30000,
    conv_approx_rank=5,
    n_processors=1,
    multi_processing=False,
    lambd=0.001,
    allowed_scale=0.01,
):
    output_dir = Path(output_dir)
    N, T, C = templates_cleaned.shape

    # Loop over templates to find residuals
    fname_spike_train_up = output_dir / "spike_train_up.npy"
    fname_spike_train = output_dir / "spike_train.npy"
    fname_templates_up = output_dir / "templates_up.npy"
    fname_scalings = output_dir / "scalings.npy"

    max_values = []
    units = []
    units_matched = []
    shifts = []

    for i in range(templates_cleaned.shape[0]):
        temp_concatenated = np.zeros((3 * T, C), dtype=np.float32)
        temp_concatenated[T : 2 * T] = templates_cleaned[i]

        # Create bin file with templates
        f_out = output_dir / "template_bin.bin"
        with open(f_out, "wb") as f:
            temp_concatenated.tofile(f)

        templates_cleaned_amputated = templates_cleaned[
            dist_argsort[i][:n_pairs_proposed]
        ]

        mp_object = MatchPursuitObjectiveUpsample(
            templates=templates_cleaned_amputated,
            deconv_dir=output_dir,
            standardized_bin=f_out,
            t_start=0,
            t_end=None,
            n_sec_chunk=1,
            sampling_rate=sampling_rate,
            max_iter=1,
            upsample=max_upsample,
            threshold=deconv_threshold,
            conv_approx_rank=conv_approx_rank,
            n_processors=n_processors,
            multi_processing=multi_processing,
            verbose=False,
            lambd=lambd,
            allowed_scale=allowed_scale,
        )

        fnames_out = []
        batch_ids = []
        for batch_id in range(mp_object.n_batches):
            fname_temp = (
                output_dir / f"seg_{str(batch_id).zfill(6)}_deconv.npz"
            )
            fnames_out.append(fname_temp)
            batch_ids.append(batch_id)

        mp_object.run(batch_ids, fnames_out)

        deconv_st = []
        deconv_scalings = []
        #         print("gathering deconvolution results")
        for batch_id in range(mp_object.n_batches):
            fname_out = output_dir / "seg_{}_deconv.npz".format(
                str(batch_id).zfill(6)
            )
            with np.load(fname_out) as d:
                deconv_st.append(d["spike_train"])
                deconv_scalings.append(d["scalings"])
        deconv_st = np.concatenate(deconv_st, axis=0)
        deconv_scalings = np.concatenate(deconv_scalings, axis=0)

        #         print(f"Number of Spikes deconvolved: {deconv_st.shape[0]}")
        if deconv_st.shape[0] > 0:
            # get spike train and save
            spike_train_tmp = np.copy(deconv_st)
            # map back to original id
            spike_train_tmp[:, 1] = np.int32(
                spike_train_tmp[:, 1] / max_upsample
            )
            spike_train_tmp[:, 0] += trough_offset
            # save
            np.save(fname_spike_train, spike_train_tmp)
            np.save(fname_scalings, deconv_scalings)

            # get upsampled templates and mapping for computing residual
            (
                templates_up,
                deconv_id_sparse_temp_map,
            ) = mp_object.get_sparse_upsampled_templates()
            np.save(fname_templates_up, templates_up.transpose(2, 0, 1))

            # get upsampled spike train
            spike_train_up = np.copy(deconv_st)
            spike_train_up[:, 1] = deconv_id_sparse_temp_map[
                spike_train_up[:, 1]
            ]
            spike_train_up[:, 0] += trough_offset
            np.save(fname_spike_train_up, spike_train_up)

            deconv_h5, deconv_residual_path = extract_deconv(
                fname_templates_up,
                fname_spike_train_up,
                output_dir,
                f_out,
                scalings_path=output_dir / "scalings.npy",
                save_cleaned_waveforms=False,
                save_denoised_waveforms=False,
                localize=False,
                n_jobs=1,
                device="cpu",
                overwrite=True,
                pbar=False,
            )

            res_array = np.fromfile(
                deconv_residual_path, dtype=np.float32
            ).reshape(-1, 384)
            max_values.append(np.abs(res_array).max())
            units.append(i)
            units_matched.append(dist_argsort[i][int(deconv_st[0, 1] / 8)])
            shifts.append(deconv_st[0, 0] - T)

    return (
        np.array(max_values),
        np.array(units),
        np.array(units_matched),
        np.array(shifts),
    )


# %%
def check_additional_merge(
    output_dir,
    temp_to_input,
    temp_to_deconv,
    deconv_threshold,
    max_upsample=8,
    trough_offset=42,
    n_pairs_proposed=10,
    sampling_rate=30000,
    conv_approx_rank=5,
    n_processors=1,
    multi_processing=False,
    lambd=0.001,
    allowed_scale=0.01,
):
    output_dir = Path(output_dir)
    T, C = temp_to_input.shape

    # Loop over templates to find residuals
    fname_spike_train_up = output_dir / "spike_train_up.npy"
    fname_spike_train = output_dir / "spike_train.npy"
    fname_templates_up = output_dir / "templates_up.npy"
    fname_scalings = output_dir / "scalings.npy"

    temp_concatenated = np.zeros((3 * T, C), dtype=np.float32)
    temp_concatenated[T : 2 * T] = temp_to_deconv

    # Create bin file with templates
    f_out = output_dir / "template_bin.bin"
    with open(f_out, "wb") as f:
        temp_concatenated.tofile(f)

    mp_object = MatchPursuitObjectiveUpsample(
        templates=temp_to_input[None, :, :],
        deconv_dir=output_dir,
        standardized_bin=f_out,
        t_start=0,
        t_end=None,
        n_sec_chunk=1,
        sampling_rate=sampling_rate,
        max_iter=1,
        upsample=max_upsample,
        threshold=deconv_threshold,
        conv_approx_rank=conv_approx_rank,
        n_processors=n_processors,
        multi_processing=multi_processing,
        verbose=False,
        lambd=lambd,
        allowed_scale=allowed_scale,
    )

    fnames_out = []
    batch_ids = []
    for batch_id in range(mp_object.n_batches):
        fname_temp = output_dir / f"seg_{str(batch_id).zfill(6)}_deconv.npz"
        fnames_out.append(fname_temp)
        batch_ids.append(batch_id)

    mp_object.run(batch_ids, fnames_out)

    deconv_st = []
    deconv_scalings = []
    for batch_id in range(mp_object.n_batches):
        fname_out = output_dir / "seg_{}_deconv.npz".format(
            str(batch_id).zfill(6)
        )
        with np.load(fname_out) as d:
            deconv_st.append(d["spike_train"])
            deconv_scalings.append(d["scalings"])
    deconv_st = np.concatenate(deconv_st, axis=0)
    deconv_scalings = np.concatenate(deconv_scalings, axis=0)

    # no spike found
    if not deconv_st.size:
        return np.inf, 0

    # get spike train and save
    spike_train_tmp = np.copy(deconv_st)
    spike_train_tmp[:, 1] = np.int32(spike_train_tmp[:, 1] / max_upsample)
    spike_train_tmp[:, 0] += trough_offset
    np.save(fname_spike_train, spike_train_tmp)
    np.save(fname_scalings, deconv_scalings)

    # get upsampled templates and mapping for computing residual
    (
        templates_up,
        deconv_id_sparse_temp_map,
    ) = mp_object.get_sparse_upsampled_templates()
    np.save(fname_templates_up, templates_up.transpose(2, 0, 1))

    # get upsampled spike train
    spike_train_up = np.copy(deconv_st)
    spike_train_up[:, 1] = deconv_id_sparse_temp_map[spike_train_up[:, 1]]
    spike_train_up[:, 0] += trough_offset
    np.save(fname_spike_train_up, spike_train_up)

    deconv_h5, deconv_residual_path = extract_deconv(
        fname_templates_up,
        fname_spike_train_up,
        output_dir,
        f_out,
        scalings_path=output_dir / "scalings.npy",
        save_cleaned_waveforms=False,
        save_denoised_waveforms=False,
        localize=False,
        n_jobs=1,
        device="cpu",
        overwrite=True,
        pbar=False,
    )

    resid = np.fromfile(deconv_residual_path, dtype=np.float32)
    return np.abs(resid).max(), deconv_st[0, 0] - T


# %%
def merge_units_temp_deconv(
    units,
    units_matched,
    max_values,
    shifts,
    templates_cleaned,
    labels,
    spike_times,
    output_dir,
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
    max_values = max_values[max_values <= merge_resid_threshold]
    shifts = shifts[max_values <= merge_resid_threshold]

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

            if (
                check_additional_merge(
                    output_dir, temp_to_input, temp_to_deconv, deconv_threshold
                )
                < merge_resid_threshold
            ):
                units_already_merged.append(matched)
                unit_reference[matched] = unit_ref

                # Update spike times
                spike_times[labels_updated == matched] -= shifts[j]
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

            if (
                check_additional_merge(
                    output_dir, temp_to_input, temp_to_deconv, deconv_threshold
                )
                < merge_resid_threshold
            ):
                units_already_merged.append(unit)
                unit_reference[unit] = unit_ref

                # Update spike times
                spike_times[labels_updated == unit] += shifts[j]
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
            if (
                check_additional_merge(
                    output_dir, temp_to_input, temp_to_deconv, deconv_threshold
                )
                < merge_resid_threshold
            ):
                unit_reference[matched] = unit_reference[unit]
                unit_reference[unit_reference[matched]] = unit_reference[unit]

                # Update spike times
                spike_times[
                    labels_updated == unit_reference[matched]
                ] -= shifts[j]
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


# %%
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
        (
            templates_updated,
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

    return labels_updated
