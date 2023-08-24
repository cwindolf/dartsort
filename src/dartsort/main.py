from pathlib import Path

from dartsort.config import FeaturizationConfig, SubtractionConfig
from dartsort.localize.localize_util import localize_hdf5
from dartsort.peel import SubtractionPeeler
from dartsort.util.data_util import DARTsortSorting, check_recording

default_featurization_config = FeaturizationConfig()
default_subtraction_config = SubtractionConfig()


def dartsort(
    recording,
    output_folder,
    *more_args_tbd,
    featurization_config=default_featurization_config,
    subtraction_config=default_subtraction_config,
    dartsort_config=...,
    n_jobs=0,
    overwrite=False,
    show_progress=True,
    device=None,
):
    # coming soon
    pass


def subtract(
    recording,
    output_directory,
    featurization_config=default_featurization_config,
    subtraction_config=default_subtraction_config,
    chunk_starts_samples=None,
    n_jobs=0,
    overwrite=False,
    residual_filename=None,
    show_progress=True,
    device=None,
    hdf5_filename="subtraction.h5",
):
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True)

    subtraction_peeler = SubtractionPeeler.from_config(
        recording,
        subtraction_config=subtraction_config,
        featurization_config=featurization_config,
    )

    # sanity check recording
    check_recording(recording)

    # fit models if needed
    model_dir = output_directory / "subtraction_models"
    if overwrite and model_dir.exists():
        for pt_file in model_dir.glob("*pipeline.pt"):
            pt_file.unlink()
    model_dir.mkdir(exist_ok=True)
    subtraction_peeler.load_or_fit_and_save_models(
        model_dir, n_jobs=n_jobs, device=device
    )

    # run main
    output_hdf5_filename = output_directory / hdf5_filename
    subtraction_peeler.peel(
        output_hdf5_filename,
        chunk_starts_samples=chunk_starts_samples,
        n_jobs=n_jobs,
        overwrite=overwrite,
        residual_filename=residual_filename,
        show_progress=show_progress,
    )

    # do localization
    if featurization_config.do_localization:
        wf_name = featurization_config.output_waveforms_name
        localize_hdf5(
            output_hdf5_filename,
            radius=featurization_config.localization_radius,
            amplitude_vectors_dataset_name=f"{wf_name}_amplitude_vectors",
            show_progress=show_progress,
            device=device,
        )

    return (
        DARTsortSorting.from_peeling_hdf5(output_hdf5_filename),
        output_hdf5_filename,
    )
