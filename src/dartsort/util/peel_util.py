from pathlib import Path

import h5py
import numpy as np
import torch

from dartsort.util.multiprocessing_util import handle_negative_jobs

from ..localize.localize_util import check_resume_or_overwrite, localize_hdf5
from ..peel.peel_base import BasePeeler
from .data_util import DARTsortSorting
from .internal_config import ComputationConfig, FeaturizationConfig
from .job_util import ensure_computation_config
from .py_util import ensure_path, timer


def run_peeler(
    peeler: BasePeeler,
    *,
    output_directory: str | Path,
    hdf5_filename: str,
    model_subdir: str,
    featurization_cfg: FeaturizationConfig,
    computation_cfg: ComputationConfig | None = None,
    chunk_starts_samples: np.ndarray | None = None,
    overwrite: bool = False,
    residual_filename: str | Path | None = None,
    skip_resid_snips: bool = False,
    show_progress: bool = True,
    fit_only: bool = False,
    stop_after_n_spikes: int | None = None,
    ensure_coverage: float | None = None,
    shuffle: bool = False,
    localization_dataset_name="point_source_localizations",
):
    output_directory = ensure_path(output_directory)
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
    computation_cfg = ensure_computation_config(computation_cfg)

    is_subsampling = stop_after_n_spikes is not None
    is_subsampling = is_subsampling and ensure_coverage != 1.0

    if peeler_is_done(
        peeler,
        output_hdf5_filename,
        overwrite=overwrite,
        chunk_starts_samples=chunk_starts_samples,
        do_localization=do_localization_later,
        localization_dataset_name=localization_dataset_name,
        stop_after_n_spikes=stop_after_n_spikes,
        ensure_coverage=ensure_coverage,
        shuffle=is_subsampling or shuffle,
    ):
        return DARTsortSorting.from_peeling_hdf5(output_hdf5_filename)

    # ensure torch linalg inits before launching threads...
    _ensure_torch_linalg(computation_cfg)

    # fit models if needed
    with timer(f"model fits ({peeler.__class__.__name__})"):
        peeler.load_or_fit_and_save_models(
            model_dir, overwrite=overwrite, computation_cfg=computation_cfg
        )
        if fit_only:
            return

    # run main
    n_resid_snips = 0 if skip_resid_snips else peeler.fit_sampling_cfg.n_residual_snips
    if is_subsampling and ensure_coverage is not None:
        n_resid_now = n_resid_snips
    elif is_subsampling:
        # can't know how many snips to extract per chunk...
        n_resid_now = 0
    else:
        n_resid_now = n_resid_snips
    if peeler.featurization_pipeline is not None:
        workers = computation_cfg.actual_n_jobs(small=True, cpu=True)
        _, workers = handle_negative_jobs(workers)
        peeler.featurization_pipeline.register_cpu_workers(workers)
    with timer(f"peel ({peeler.__class__.__name__})"):
        peeler.peel(
            output_hdf5_filename,
            chunk_starts_samples=chunk_starts_samples,
            overwrite=overwrite,
            residual_filename=residual_filename,
            show_progress=show_progress,
            computation_cfg=computation_cfg,
            total_residual_snips=n_resid_now,
            stop_after_n_waveforms=stop_after_n_spikes,
            ensure_coverage=ensure_coverage,
            shuffle=is_subsampling or shuffle,
        )
    if n_resid_now == 0 and n_resid_snips > 0:
        with timer(f"residuals ({peeler.__class__.__name__})"):
            peeler.run_subsampled_peeling(
                output_hdf5_filename,
                chunk_length_samples=peeler.spike_length_samples,
                residual_to_h5=True,
                skip_features=True,
                ignore_resuming=True,
                computation_cfg=computation_cfg,
                n_chunks=n_resid_snips,
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
    stop_after_n_spikes: int | None = None,
    ensure_coverage: float | None = None,
    do_localization=True,
    localization_dataset_name="point_source_localizations",
    main_channels_dataset_name="channels",
    shuffle=False,
):
    if overwrite:
        return False

    if not output_hdf5_filename.exists():
        return False

    if n_residual_snips:
        with h5py.File(output_hdf5_filename, "r") as h5:
            if "residual" not in h5:
                return False
            nr = h5["n_residuals"][()]
            if h5["residual"].chunks is not None:
                assert nr == h5["residual"].shape[0]
            if nr < n_residual_snips:
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

    chunk_starts_samples = peeler.get_chunk_starts(
        chunk_starts_samples=chunk_starts_samples, subsampled=shuffle, n_chunks=np.inf
    )
    done, *_ = peeler.check_resuming(
        output_hdf5_filename,
        chunk_starts_samples=chunk_starts_samples,
        stop_after_n_waveforms=stop_after_n_spikes,
        ensure_coverage=ensure_coverage,
        overwrite=False,
    )
    return done


def _ensure_torch_linalg(computation_cfg):
    device = computation_cfg.actual_device()
    if device.type == "cpu":
        return
    if computation_cfg.actual_n_jobs() == 0:
        return

    torch.cuda.synchronize()
    torch.inverse(torch.ones((0, 0), device=device))
