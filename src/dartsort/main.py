from pathlib import Path

from dartsort.config import (default_featurization_config,
                             default_matching_config,
                             default_subtraction_config,
                             default_template_config)
from dartsort.localize.localize_util import localize_hdf5
from dartsort.peel import (ObjectiveUpdateTemplateMatchingPeeler,
                           SubtractionPeeler)
from dartsort.templates import TemplateData
from dartsort.util.data_util import DARTsortSorting, check_recording


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
    model_subdir="subtraction_models",
):
    check_recording(recording)
    subtraction_peeler = SubtractionPeeler.from_config(
        recording,
        subtraction_config=subtraction_config,
        featurization_config=featurization_config,
    )
    detections, output_hdf5_filename = _run_peeler(
        subtraction_peeler,
        output_directory,
        hdf5_filename,
        model_subdir,
        featurization_config,
        chunk_starts_samples=chunk_starts_samples,
        overwrite=overwrite,
        n_jobs=n_jobs,
        residual_filename=residual_filename,
        show_progress=show_progress,
        device=device,
    )
    return detections, output_hdf5_filename


def cluster(*args):
    # coming soon
    pass


def match(
    recording,
    sorting,
    output_directory,
    motion_est=None,
    template_config=default_template_config,
    featurization_config=default_featurization_config,
    matching_config=default_matching_config,
    chunk_starts_samples=None,
    n_jobs_templates=0,
    n_jobs_match=0,
    overwrite=False,
    residual_filename=None,
    show_progress=True,
    device=None,
    hdf5_filename="matching0.h5",
    model_subdir="matching0_models",
    template_npz_filename="template_data.npz",
):
    model_dir = Path(output_directory) / model_subdir

    # compute templates
    template_data = TemplateData.from_config(
        recording,
        sorting,
        template_config,
        motion_est=motion_est,
        n_jobs=n_jobs_templates,
        save_folder=model_dir,
        overwrite=overwrite,
        device=device,
        save_npz_name=template_npz_filename,
    )

    # instantiate peeler
    matching_peeler = ObjectiveUpdateTemplateMatchingPeeler.from_config(
        recording,
        matching_config,
        featurization_config,
        template_data,
        motion_est=motion_est,
    )
    sorting, output_hdf5_filename = _run_peeler(
        matching_peeler,
        output_directory,
        hdf5_filename,
        model_subdir,
        featurization_config,
        chunk_starts_samples=chunk_starts_samples,
        overwrite=overwrite,
        n_jobs=n_jobs_match,
        residual_filename=residual_filename,
        show_progress=show_progress,
        device=device,
    )
    return sorting, output_hdf5_filename


# -- helper function for subtract, match


def _run_peeler(
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
):
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True)
    model_dir = output_directory / model_subdir
    output_hdf5_filename = output_directory / hdf5_filename
    if residual_filename is not None:
        residual_filename = output_directory / residual_filename

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
    del peeler
    _gc(n_jobs, device)

    # do localization
    if not featurization_config.denoise_only and featurization_config.do_localization:
        wf_name = featurization_config.output_waveforms_name
        localize_hdf5(
            output_hdf5_filename,
            radius=featurization_config.localization_radius,
            amplitude_vectors_dataset_name=f"{wf_name}_amplitude_vectors",
            show_progress=show_progress,
            device=device,
            localization_model=featurization_config.localization_model
        )
        _gc(0, device)

    return (
        DARTsortSorting.from_peeling_hdf5(output_hdf5_filename),
        output_hdf5_filename,
    )

def _gc(n_jobs, device):
    if n_jobs:
        # work happened off main process
        return

    import gc, torch
    gc.collect()
    
    if (
        torch.device(device).type == "cuda"
        or (torch.cuda.is_available() and device is None)
    ):
        torch.cuda.empty_cache()
