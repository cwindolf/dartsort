from pathlib import Path

import h5py
import numpy as np
import torch

from ..localize.localize_util import check_resume_or_overwrite, localize_hdf5
from .data_util import DARTsortSorting, batched_h5_read


def run_peeler(
    peeler,
    output_directory,
    hdf5_filename,
    model_subdir,
    featurization_config,
    chunk_starts_samples=None,
    overwrite=False,
    n_jobs=0,
    residual_filename=None,
    show_progress=True,
    device=None,
    localization_dataset_name="point_source_localizations",
):
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True)
    model_dir = output_directory / model_subdir
    output_hdf5_filename = output_directory / hdf5_filename
    if residual_filename is not None:
        residual_filename = output_directory / residual_filename
    do_localization = (
        not featurization_config.denoise_only
        and featurization_config.do_localization
        and not featurization_config.nn_localization
    )

    if peeler_is_done(
        peeler,
        output_hdf5_filename,
        overwrite=overwrite,
        chunk_starts_samples=chunk_starts_samples,
        do_localization=do_localization,
        localization_dataset_name=localization_dataset_name,
    ):
        return (
            DARTsortSorting.from_peeling_hdf5(output_hdf5_filename),
            output_hdf5_filename,
        )

    # fit models if needed
    peeler.load_or_fit_and_save_models(
        model_dir, overwrite=overwrite, n_jobs=n_jobs, device=device
    )

    # run main
    peeler.peel(
        output_hdf5_filename,
        chunk_starts_samples=chunk_starts_samples,
        n_jobs=n_jobs,
        overwrite=overwrite,
        residual_filename=residual_filename,
        show_progress=show_progress,
        device=device,
    )
    _gc(n_jobs, device)

    # do localization
    if do_localization:
        wf_name = featurization_config.output_waveforms_name
        loc_amp_type = featurization_config.localization_amplitude_type
        localize_hdf5(
            output_hdf5_filename,
            radius=featurization_config.localization_radius,
            amplitude_vectors_dataset_name=f"{wf_name}_{loc_amp_type}_amplitude_vectors",
            output_dataset_name=localization_dataset_name,
            show_progress=show_progress,
            n_jobs=n_jobs,
            device=device,
            localization_model=featurization_config.localization_model,
        )
        _gc(n_jobs, device)

    if featurization_config.n_residual_snips:
        peeler.run_subsampled_peeling(
            output_hdf5_filename,
            n_jobs=n_jobs,
            chunk_length_samples=peeler.spike_length_samples,
            residual_to_h5=True,
            skip_features=True,
            ignore_resuming=True,
            device=device,
            n_chunks=featurization_config.n_residual_snips,
            task_name="Residual snips",
            overwrite=False,
            ordered=True,
            skip_last=True,
        )

    return (
        DARTsortSorting.from_peeling_hdf5(output_hdf5_filename),
        output_hdf5_filename,
    )


def peeler_is_done(
    peeler,
    output_hdf5_filename,
    overwrite=False,
    n_residual_snips=0,
    chunk_starts_samples=None,
    do_localization=True,
    localization_dataset_name="point_source_localizations",
    main_channels_dataset_name="channels",
):
    if overwrite:
        return False

    if not output_hdf5_filename.exists():
        return False

    if n_residual_snips:
        with h5py.File(output_hdf5_filename, "r") as h5:
            if "residual" not in h5:
                return False
            if len(h5["residual"]) < n_residual_snips:
                return False

    if do_localization:
        (
            done,
            output_hdf5_filename,
            next_batch_start,
        ) = check_resume_or_overwrite(
            output_hdf5_filename,
            localization_dataset_name,
            main_channels_dataset_name,
            overwrite=False,
        )
        return done

    last_chunk_start = peeler.check_resuming(
        output_hdf5_filename,
        overwrite=False,
    )
    chunk_starts_samples = peeler.get_chunk_starts(
        chunk_starts_samples=chunk_starts_samples
    )
    return last_chunk_start >= max(chunk_starts_samples)


def _gc(n_jobs, device):
    if n_jobs:
        # work happened off main process
        return

    import gc
    import torch

    gc.collect()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.device(device).type == "cuda" or (
        torch.cuda.is_available() and device is None
    ):
        torch.cuda.empty_cache()


def subsample_waveforms(
    hdf5_filename,
    fit_sampling="random",
    random_state=0,
    n_waveforms_fit=10_000,
    voltages_dataset_name="collisioncleaned_voltages",
    waveforms_dataset_name="collisioncleaned_waveforms",
    fit_max_reweighting=20.0,
):
    from ..cluster.density import get_smoothed_densities
    random_state = np.random.default_rng(random_state)

    with h5py.File(hdf5_filename) as h5:
        channels = h5["channels"][:]
        n_wf = channels.size
        if n_wf > n_waveforms_fit:
            if fit_sampling == "random":
                choices = random_state.choice(
                    n_wf, size=n_waveforms_fit, replace=False
                )
            elif fit_sampling == "amp_reweighted":
                volts = h5[voltages_dataset_name][:]
                sigma = 1.06 * volts.std() * np.power(len(volts), -0.2)
                sample_p = get_smoothed_densities(volts[:, None], sigmas=sigma)
                sample_p = sample_p.mean() / sample_p
                sample_p = sample_p.clip(
                    1. / fit_max_reweighting,
                    fit_max_reweighting,
                )
                sample_p /= sample_p.sum()
                choices = random_state.choice(
                    n_wf, p=sample_p, size=n_waveforms_fit, replace=False
                )
            else:
                assert False
            choices.sort()
            channels = channels[choices]
            waveforms = batched_h5_read(
                h5[waveforms_dataset_name], choices
            )
        else:
            waveforms = h5[waveforms_dataset_name][:]

    waveforms = torch.from_numpy(waveforms)
    channels = torch.from_numpy(channels)

    return channels, waveforms
