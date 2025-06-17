from pathlib import Path

import h5py

from ..localize.localize_util import check_resume_or_overwrite, localize_hdf5
from .data_util import DARTsortSorting
from . import job_util
from .py_util import resolve_path
from ..peel.peel_base import BasePeeler
from .internal_config import FeaturizationConfig, ComputationConfig


def run_peeler(
    peeler: BasePeeler,
    output_directory: str | Path,
    hdf5_filename: str,
    model_subdir: str,
    featurization_cfg: FeaturizationConfig,
    computation_cfg: ComputationConfig | None = None,
    chunk_starts_samples=None,
    overwrite=False,
    residual_filename: str | Path | None = None,
    show_progress=True,
    fit_only=False,
    localization_dataset_name="point_source_localizations",
):
    output_directory = resolve_path(output_directory)
    output_directory.mkdir(exist_ok=True)
    model_dir = output_directory / model_subdir
    output_hdf5_filename = output_directory / hdf5_filename
    if residual_filename is not None:
        if not isinstance(residual_filename, Path):
            residual_filename = output_directory / residual_filename
    do_localization_later = (
        not featurization_cfg.denoise_only
        and featurization_cfg.do_localization
        and not featurization_cfg.nn_localization
    )
    if computation_cfg is None:
        computation_cfg = job_util.get_global_computation_config()

    if peeler_is_done(
        peeler,
        output_hdf5_filename,
        overwrite=overwrite,
        chunk_starts_samples=chunk_starts_samples,
        do_localization=do_localization_later,
        localization_dataset_name=localization_dataset_name,
    ):
        return DARTsortSorting.from_peeling_hdf5(output_hdf5_filename)

    # fit models if needed
    peeler.load_or_fit_and_save_models(
        model_dir, overwrite=overwrite, computation_cfg=computation_cfg
    )
    if fit_only:
        return

    # run main
    n_resid_now = featurization_cfg.n_residual_snips * int(
        not featurization_cfg.residual_later
    )
    peeler.peel(
        output_hdf5_filename,
        chunk_starts_samples=chunk_starts_samples,
        overwrite=overwrite,
        residual_filename=residual_filename,
        show_progress=show_progress,
        computation_cfg=computation_cfg,
        total_residual_snips=n_resid_now,
        stop_after_n_waveforms=featurization_cfg.stop_after_n,
        shuffle=featurization_cfg.shuffle,
    )

    if featurization_cfg.residual_later:
        peeler.run_subsampled_peeling(
            output_hdf5_filename,
            chunk_length_samples=peeler.spike_length_samples,
            residual_to_h5=True,
            skip_features=True,
            ignore_resuming=True,
            computation_cfg=computation_cfg,
            n_chunks=featurization_cfg.n_residual_snips,
            task_name="Residual snips",
            overwrite=False,
            ordered=True,
            skip_last=True,
        )

    # do localization
    if do_localization_later:
        wf_name = featurization_cfg.output_waveforms_name
        loc_amp_type = featurization_cfg.localization_amplitude_type
        localize_hdf5(
            output_hdf5_filename,
            radius=featurization_cfg.localization_radius,
            amplitude_vectors_dataset_name=f"{wf_name}_{loc_amp_type}_amplitude_vectors",
            output_dataset_name=localization_dataset_name,
            show_progress=show_progress,
            n_jobs=computation_cfg.actual_n_jobs(),
            device=computation_cfg.actual_device(),
            localization_model=featurization_cfg.localization_model,
        )

    return DARTsortSorting.from_peeling_hdf5(output_hdf5_filename)


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

    last_chunk_start, n_residual_snips = peeler.check_resuming(
        output_hdf5_filename,
        overwrite=False,
    )
    chunk_starts_samples = peeler.get_chunk_starts(
        chunk_starts_samples=chunk_starts_samples
    )
    return last_chunk_start >= max(chunk_starts_samples)
