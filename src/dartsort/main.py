from dataclasses import asdict
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from spikeinterface.core import BaseRecording

from dartsort.cluster.initial import initial_clustering
from dartsort.cluster.refine import refine_clustering
from dartsort.config import (
    DARTsortUserConfig,
    DARTsortInternalConfig,
    DeveloperConfig,
    to_internal_config,
    default_dartsort_config,
    default_waveform_config,
    default_template_config,
    default_clustering_config,
    default_featurization_config,
    default_subtraction_config,
    default_matching_config,
    default_thresholding_config,
    default_computation_config,
)
from dartsort.peel import (
    ObjectiveUpdateTemplateMatchingPeeler,
    SubtractionPeeler,
    GrabAndFeaturize,
    ThresholdAndFeaturize,
)
from dartsort.templates import TemplateData
from dartsort.util.data_util import (
    DARTsortSorting,
    check_recording,
    keep_only_most_recent_spikes,
)
from dartsort.util.peel_util import run_peeler
from dartsort.util.registration_util import estimate_motion
from dartsort.util.py_util import resolve_path
from dartsort.util.main_util import (
    ds_save_intermediate_labels,
    ds_all_to_workdir,
    ds_save_motion_est,
    ds_save_intermediate_features,
    ds_handle_delete_intermediate_features,
)


logger = getLogger(__name__)


def dartsort(
    recording: BaseRecording,
    output_dir: str | Path,
    cfg: (
        DARTsortUserConfig | DeveloperConfig | DARTsortInternalConfig
    ) = default_dartsort_config,
    motion_est=None,
    overwrite=False,
):
    """TODO: fast forward."""
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(exist_ok=True)
    cfg = to_internal_config(cfg)
    if cfg.work_in_tmpdir:
        with TemporaryDirectory(prefix="dartsort", dir=cfg.tmpdir_parent) as work_dir:
            work_dir = resolve_path(work_dir)
            ds_all_to_workdir(output_dir, work_dir, overwrite)
            return _dartsort_impl(
                recording, output_dir, cfg, motion_est, work_dir, overwrite
            )
    return _dartsort_impl(recording, output_dir, cfg, motion_est, None, overwrite)


def _dartsort_impl(
    recording: BaseRecording,
    output_dir: Path,
    cfg: DARTsortInternalConfig = default_dartsort_config,
    motion_est=None,
    work_dir=None,
    overwrite=False,
):
    ret = {}

    store_dir = output_dir if work_dir is None else work_dir

    # first step: initial detection and motion estimation
    sorting = subtract(
        output_dir=store_dir,
        recording=recording,
        waveform_config=cfg.waveform_config,
        featurization_config=cfg.featurization_config,
        subtraction_config=cfg.subtraction_config,
        computation_config=cfg.computation_config,
        overwrite=overwrite,
    )
    logger.info(f"Initial detection: {sorting}")
    ds_save_intermediate_features(cfg, sorting, output_dir, work_dir)

    if cfg.subtract_only:
        ret["sorting"] = sorting
        return ret

    if motion_est is None:
        motion_est = estimate_motion(
            output_directory=store_dir,
            recording=recording,
            sorting=sorting,
            overwrite=overwrite,
            device=cfg.computation_config.actual_device(),
            **asdict(cfg.motion_estimation_config),
        )
    ret["motion_est"] = motion_est
    ds_save_motion_est(motion_est, output_dir, work_dir, overwrite)

    if cfg.dredge_only:
        ret["sorting"] = sorting
        return ret

    # clustering: initialization
    sorting = initial_clustering(
        recording=recording,
        sorting=sorting,
        motion_est=motion_est,
        clustering_config=cfg.clustering_config,
        computation_config=cfg.computation_config,
    )
    logger.info(f"Initial clustering: {sorting}")
    ds_save_intermediate_labels("initial", sorting, output_dir, cfg)

    # clustering: model
    sdir = sfmt = None
    if cfg.save_intermediate_labels:
        sdir = output_dir
        sfmt = "refined0{stepname}"
    sorting, _ = refine_clustering(
        recording=recording,
        sorting=sorting,
        motion_est=motion_est,
        refinement_config=cfg.initial_refinement_config,
        computation_config=cfg.computation_config,
        save_step_labels_format=sfmt,
        save_step_labels_dir=sdir,
        save_cfg=cfg,
    )
    logger.info(f"Initial refinement: {sorting}")
    ds_save_intermediate_labels("refined0", sorting, output_dir, cfg)

    for step in range(1, cfg.matching_iterations + 1):
        is_final = step == cfg.matching_iterations

        # TODO
        # prop = 1.0 if is_final else cfg.intermediate_matching_subsampling

        sorting = match(
            output_dir=store_dir,
            recording=recording,
            sorting=sorting,
            motion_est=motion_est,
            template_config=cfg.template_config,
            waveform_config=cfg.waveform_config,
            featurization_config=cfg.featurization_config,
            matching_config=cfg.matching_config,
            overwrite=overwrite or cfg.overwrite_matching,
            computation_config=cfg.computation_config,
            hdf5_filename=f"matching{step}.h5",
            model_subdir=f"matching{step}_models",
        )
        logger.info(f"Matching step {step}: {sorting}")
        ds_save_intermediate_labels(f"matching{step}", sorting, output_dir, cfg)
        if not is_final:
            ds_save_intermediate_features(cfg, sorting, output_dir, work_dir)

        if cfg.final_refinement or not is_final:
            sdir = sfmt = None
            if cfg.save_intermediate_labels:
                sdir = output_dir
                sfmt = f"refined{step}{{stepname}}"
            sorting, _ = refine_clustering(
                recording=recording,
                sorting=sorting,
                motion_est=motion_est,
                refinement_config=cfg.refinement_config,
                computation_config=cfg.computation_config,
                save_step_labels_format=sfmt,
                save_step_labels_dir=sdir,
                save_cfg=cfg,
            )
            logger.info(f"Refinement step {step}: {sorting}")
            ds_save_intermediate_labels(f"refined{step}", sorting, output_dir, cfg)

    ds_handle_delete_intermediate_features(cfg, sorting, output_dir, work_dir)

    sorting.save(output_dir / "dartsort_sorting.npz")
    ret["sorting"] = sorting
    return ret


def subtract(
    output_dir: str | Path,
    recording: BaseRecording,
    waveform_config=default_waveform_config,
    featurization_config=default_featurization_config,
    subtraction_config=default_subtraction_config,
    computation_config=default_computation_config,
    chunk_starts_samples=None,
    overwrite=False,
    residual_filename: str | None = None,
    show_progress=True,
    hdf5_filename="subtraction.h5",
    model_subdir="subtraction_models",
) -> DARTsortSorting:
    output_dir = resolve_path(output_dir)
    check_recording(recording)
    subtraction_peeler = SubtractionPeeler.from_config(
        recording,
        waveform_config=waveform_config,
        subtraction_config=subtraction_config,
        featurization_config=featurization_config,
    )
    detections = run_peeler(
        subtraction_peeler,
        output_dir,
        hdf5_filename,
        model_subdir=model_subdir,
        featurization_config=featurization_config,
        chunk_starts_samples=chunk_starts_samples,
        overwrite=overwrite,
        computation_config=computation_config,
        residual_filename=residual_filename,
        show_progress=show_progress,
    )
    return detections


def match(
    output_dir: str | Path,
    recording: BaseRecording,
    sorting: DARTsortSorting | None = None,
    motion_est=None,
    waveform_config=default_waveform_config,
    template_config=default_template_config,
    featurization_config=default_featurization_config,
    matching_config=default_matching_config,
    chunk_starts_samples=None,
    overwrite=False,
    residual_filename: str | None = None,
    show_progress=True,
    hdf5_filename="matching0.h5",
    model_subdir="matching0_models",
    template_data: TemplateData | None = None,
    template_npz_filename="template_data.npz",
    computation_config=default_computation_config,
) -> DARTsortSorting:
    output_dir = resolve_path(output_dir)
    model_dir = output_dir / model_subdir

    # compute templates
    if template_data is None:
        template_data = TemplateData.from_config(
            recording,
            sorting,
            template_config=template_config,
            waveform_config=waveform_config,
            motion_est=motion_est,
            save_folder=model_dir,
            overwrite=overwrite,
            save_npz_name=template_npz_filename,
            computation_config=computation_config,
            with_locs=motion_est is not None,
        )

    # instantiate peeler
    matching_peeler = ObjectiveUpdateTemplateMatchingPeeler.from_config(
        recording,
        waveform_config,
        matching_config,
        featurization_config,
        template_data,
        motion_est=motion_est,
    )
    sorting = run_peeler(
        matching_peeler,
        output_dir,
        hdf5_filename,
        model_subdir,
        featurization_config,
        chunk_starts_samples=chunk_starts_samples,
        overwrite=overwrite,
        residual_filename=residual_filename,
        show_progress=show_progress,
        computation_config=computation_config,
    )
    return sorting


def grab(
    output_dir: str | Path,
    recording: BaseRecording,
    sorting: DARTsortSorting,
    waveform_config=default_waveform_config,
    featurization_config=default_featurization_config,
    chunk_starts_samples=None,
    overwrite=False,
    show_progress=True,
    hdf5_filename="grab.h5",
    model_subdir="grab_models",
    computation_config=default_computation_config,
) -> DARTsortSorting:
    output_dir = resolve_path(output_dir)
    grabber = GrabAndFeaturize.from_config(
        sorting,
        recording,
        waveform_config,
        featurization_config,
    )
    sorting = run_peeler(
        grabber,
        output_dir,
        hdf5_filename,
        model_subdir,
        featurization_config,
        chunk_starts_samples=chunk_starts_samples,
        overwrite=overwrite,
        show_progress=show_progress,
        computation_config=computation_config,
    )
    return sorting


def threshold(
    output_dir: str | Path,
    recording: BaseRecording,
    waveform_config=default_waveform_config,
    thresholding_config=default_thresholding_config,
    featurization_config=default_featurization_config,
    chunk_starts_samples=None,
    overwrite=False,
    show_progress=True,
    hdf5_filename="threshold.h5",
    model_subdir="threshold_models",
    computation_config=default_computation_config,
) -> DARTsortSorting:
    output_dir = resolve_path(output_dir)
    thresholder = ThresholdAndFeaturize.from_config(
        recording, waveform_config, thresholding_config, featurization_config
    )
    sorting = run_peeler(
        thresholder,
        output_dir,
        hdf5_filename,
        model_subdir,
        featurization_config,
        chunk_starts_samples=chunk_starts_samples,
        overwrite=overwrite,
        show_progress=show_progress,
        computation_config=computation_config,
    )
    return sorting


def match_chunked(
    recording,
    sorting,
    output_dir=None,
    motion_est=None,
    waveform_config=default_waveform_config,
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
    template_data=None,
    template_npz_filename="template_data.npz",
):
    # compute chunk time ranges
    chunk_samples = recording.sampling_frequency * template_config.chunk_size_s
    n_chunks = recording.get_num_samples() / chunk_samples
    # we'll count the remainder as a chunk if it's at least 2/3 of one
    n_chunks = np.floor(n_chunks) + (n_chunks - np.floor(n_chunks) > 0.66)
    n_chunks = int(max(1, n_chunks))

    # evenly divide the recording into chunks
    assert recording.get_num_segments() == 1
    start_time_s, end_time_s = recording._recording_segments[0].sample_index_to_time(
        np.array([0, recording.get_num_samples() - 1])
    )
    chunk_times_s = np.linspace(start_time_s, end_time_s, num=n_chunks + 1)
    chunk_time_ranges_s = list(zip(chunk_times_s[:-1], chunk_times_s[1:]))

    sortings = []
    hdf5_filenames = []

    for j, chunk_time_range in enumerate(chunk_time_ranges_s):
        sorting_chunk = keep_only_most_recent_spikes(
            sorting,
            n_min_spikes=template_config.spikes_per_unit,
            latest_time_sample=chunk_time_range[1] * recording.sampling_frequency,
        )
        chunk_starts_samples = recording._recording_segments[0].time_to_sample_index(
            chunk_time_range
        )
        chunk_starts_samples = chunk_starts_samples.astype(int)
        chunk_starts_samples = np.arange(
            *chunk_starts_samples, matching_config.chunk_length_samples
        )

        chunk_sorting, chunk_h5 = match(
            recording,
            sorting=sorting_chunk,
            output_dir=output_dir,
            motion_est=motion_est,
            waveform_config=default_waveform_config,
            template_config=default_template_config,
            featurization_config=default_featurization_config,
            matching_config=default_matching_config,
            chunk_starts_samples=chunk_starts_samples,
            n_jobs_templates=n_jobs_templates,
            n_jobs_match=n_jobs_match,
            overwrite=overwrite,
            residual_filename=None,
            show_progress=show_progress,
            device=device,
            hdf5_filename=f"matching0_chunk{j:3d}.h5",
            model_subdir=f"matching0_chunk{j:3d}_models",
            template_npz_filename=template_npz_filename,
        )

        sortings.append(chunk_sorting)
        hdf5_filenames.append(chunk_h5)

    return sortings, hdf5_filenames
