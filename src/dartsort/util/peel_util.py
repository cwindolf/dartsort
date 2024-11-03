from pathlib import Path

from ..localize.localize_util import check_resume_or_overwrite, localize_hdf5
from .data_util import DARTsortSorting
from dartsort.peel import ObjectiveUpdateTemplateMatchingPeeler

from dartsort.config import (
    default_featurization_config,
    default_matching_config,
    default_waveform_config,
)

def fit_and_save_models(
    peeler,
    output_directory,
    hdf5_filename,
    model_subdir,
    featurization_config,
    chunk_starts_samples=None,
    subsampling_proportion=1.0,
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
    )

    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True)
    model_dir = output_directory / model_subdir
    output_hdf5_filename = output_directory / hdf5_filename
    if residual_filename is not None:
        residual_filename = output_directory / residual_filename
    do_localization = (
        not featurization_config.denoise_only
        and featurization_config.do_localization
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

    del peeler
    _gc(n_jobs, device)


def run_peeler(
    peeler,
    output_directory,
    hdf5_filename,
    model_subdir,
    featurization_config,
    chunk_starts_samples=None,
    subsampling_proportion=1.0,
    overwrite=False,
    exception_no_featurization=False,
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
    )

    # if not keep_writing:
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
        model_dir, overwrite=overwrite, exception_no_featurization=exception_no_featurization, n_jobs=n_jobs, device=device
    )

    # run main
    if chunk_starts_samples is None and subsampling_proportion < 1.0:
        chunk_starts_samples = peeler.get_chunk_starts(
            subsampling_proportion=subsampling_proportion
        )
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

    return (
        DARTsortSorting.from_peeling_hdf5(output_hdf5_filename),
        output_hdf5_filename,
    )



def match_single_chunk(
    recording,
    template_data,
    name_hdf5_chunk="matching0.h5",
    chunk_starts_samples=None,
    motion_est=None,
    output_directory=None,
    model_subdir="matching0_models",
    waveform_config=default_waveform_config,
    featurization_config=default_featurization_config,
    matching_config=default_matching_config,
    n_jobs=0,
    overwrite=False,
    residual_filename=None,
    show_progress=True,
    device=None,
):
    matching_peeler = ObjectiveUpdateTemplateMatchingPeeler.from_config(
        recording,
        waveform_config,
        matching_config,
        featurization_config,
        template_data,
        motion_est=motion_est,
    )
    sorting_chunk, output_hdf5_filename = run_peeler(
        matching_peeler,
        output_directory,
        name_hdf5_chunk,
        model_subdir,
        featurization_config,
        chunk_starts_samples=chunk_starts_samples,
        overwrite=overwrite,
        exception_no_featurization=True,
        n_jobs=n_jobs,
        residual_filename=residual_filename,
        show_progress=show_progress,
        device=device,
    )
    return sorting_chunk, output_hdf5_filename

def fitting_and_saving_models_before_time_tracking_deconv(
    recording,
    template_data,
    hdf5_filename="matching0.h5",
    chunk_starts_samples=None,
    motion_est=None,
    output_directory=None,
    model_subdir="matching0_models",
    waveform_config=default_waveform_config,
    featurization_config=default_featurization_config,
    matching_config=default_matching_config,
    n_jobs=0,
    overwrite=True,
    residual_filename=None,
    show_progress=True,
    device=None,
):
       
    matching_peeler = ObjectiveUpdateTemplateMatchingPeeler.from_config(
        recording,
        waveform_config,
        matching_config,
        featurization_config,
        template_data, 
        motion_est=motion_est,
    )
    
    fit_and_save_models(
        matching_peeler,
        output_directory,
        hdf5_filename,
        model_subdir,
        featurization_config,
        chunk_starts_samples=chunk_starts_samples,
        overwrite=True,
        n_jobs=n_jobs,
        residual_filename=residual_filename,
        show_progress=show_progress,
        device=device,
    )



def peeler_is_done(
    peeler,
    output_hdf5_filename,
    overwrite=False,
    chunk_starts_samples=None,
    do_localization=True,
    localization_dataset_name="point_source_localizations",
    main_channels_dataset_name="channels",
):
    if overwrite:
        return False

    if not output_hdf5_filename.exists():
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
