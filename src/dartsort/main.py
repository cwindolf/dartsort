from dataclasses import asdict, replace
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
import traceback
import shutil

import numpy as np
from spikeinterface.core import BaseRecording

from .cluster import (
    get_clusterer,
    get_clustering_features,
    postprocess,
)
from .config import DARTsortUserConfig, DeveloperConfig
from .peel import (
    ObjectiveUpdateTemplateMatchingPeeler,
    SubtractionPeeler,
    GrabAndFeaturize,
    ThresholdAndFeaturize,
    UniversalTemplatesMatchingPeeler,
)
from .templates import TemplateData
from .util.data_util import (
    DARTsortSorting,
    check_recording,
    keep_only_most_recent_spikes,
)
from .util.internal_config import (
    DARTsortInternalConfig,
    ComputationConfig,
    ClusteringConfig,
    ClusteringFeaturesConfig,
    RefinementConfig,
    SubtractionConfig,
    ThresholdingConfig,
    MatchingConfig,
    UniversalMatchingConfig,
    to_internal_config,
    default_dartsort_cfg,
    default_waveform_cfg,
    default_template_cfg,
    default_clustering_cfg,
    default_clustering_features_cfg,
    default_featurization_cfg,
    default_subtraction_cfg,
    default_matching_cfg,
    default_thresholding_cfg,
    default_universal_cfg,
)
from .util.logging_util import DARTsortLogger
from .util.main_util import (
    ds_all_to_workdir,
    ds_dump_config,
    ds_fast_forward,
    ds_handle_delete_intermediate_features,
    ds_save_features,
    ds_save_motion_est,
)
from .util.peel_util import run_peeler
from .util.py_util import resolve_path
from .util.registration_util import estimate_motion


logger: DARTsortLogger = getLogger(__name__)


def dartsort(
    recording: BaseRecording,
    output_dir: str | Path,
    cfg: (
        DARTsortUserConfig | DeveloperConfig | DARTsortInternalConfig
    ) = default_dartsort_cfg,
    motion_est=None,
    overwrite=False,
    allow_symlinks=True,
):
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # convert cfg to internal format and store it for posterity
    cfg = to_internal_config(cfg)
    ds_dump_config(cfg, output_dir)

    if cfg.work_in_tmpdir:
        with TemporaryDirectory(prefix="dartsort", dir=cfg.tmpdir_parent) as work_dir:
            work_dir = resolve_path(work_dir)
            logger.dartsortdebug(f"Working in {work_dir}, outputs to {output_dir}.")
            ds_all_to_workdir(output_dir, work_dir, overwrite, allow_symlinks)
            try:
                return _dartsort_impl(
                    recording, output_dir, cfg, motion_est, work_dir, overwrite
                )
            except Exception as e:
                traceback_path = output_dir / "traceback.txt"
                error_data_path = output_dir / "error_state"
                with open(traceback_path, "w") as f:
                    traceback.print_exception(e, file=f)
                if cfg.save_everything_on_error:
                    logger.critical(
                        f"Hit an error. Copying outputs to {error_data_path} "
                        f"and writing traceback to {traceback_path}."
                    )
                    shutil.copytree(work_dir, error_data_path, dirs_exist_ok=True)
                else:
                    logger.critical(
                        f"Hit an error. Writing traceback to {traceback_path}."
                        " work_in_tmpdir was true, so the files won't be kept."
                        " Set save_everything_on_error to keep them."
                    )
                raise

    try:
        return _dartsort_impl(recording, output_dir, cfg, motion_est, None, overwrite)
    except Exception as e:
        traceback_path = output_dir / "traceback.txt"
        with open(traceback_path, "w") as f:
            traceback.print_exception(e, file=f)
        logger.critical(f"Hit an error. Wrote traceback to {traceback_path}.")
        raise


def _dartsort_impl(
    recording: BaseRecording,
    output_dir: Path,
    cfg: DARTsortInternalConfig = default_dartsort_cfg,
    motion_est=None,
    work_dir=None,
    overwrite=False,
):
    ret = {}

    store_dir = output_dir if work_dir is None else work_dir
    next_step, sorting = ds_fast_forward(store_dir, cfg)

    if next_step == 0:
        # first step: initial detection and motion estimation
        sorting = initial_detection(
            output_dir=store_dir,
            recording=recording,
            cfg=cfg,
            overwrite=overwrite,
            motion_est=motion_est,
        )
        assert sorting is not None
        logger.info(f"Initial detection: {sorting}")
        is_final = cfg.detect_only or cfg.dredge_only or not cfg.matching_iterations
        ds_save_features(cfg, sorting, output_dir, work_dir, is_final=is_final)

        if cfg.detect_only:
            ret["sorting"] = sorting
            return ret

        if motion_est is None:
            logger.dartsortdebug("-- Estimate motion")
            motion_est = estimate_motion(
                output_directory=store_dir,
                recording=recording,
                sorting=sorting,
                overwrite=overwrite,
                device=cfg.computation_cfg.actual_device(),
                **asdict(cfg.motion_estimation_cfg),
            )
        ret["motion_est"] = motion_est
        ds_save_motion_est(motion_est, output_dir, work_dir, overwrite)

        if cfg.dredge_only:
            ret["sorting"] = sorting
            return ret

        # clustering: initialization and first refinement
        sorting = cluster(
            recording,
            sorting,
            motion_est=motion_est,
            refinement_cfg=cfg.initial_refinement_cfg,
            clustering_cfg=cfg.clustering_cfg,
            clustering_features_cfg=cfg.clustering_features_cfg,
            _save_cfg=cfg,
            _save_dir=output_dir,
        )
        logger.info(f"First clustering: {sorting}")

        # be sure to start matching at step 1
        next_step += 1
    assert sorting is not None
    assert next_step > 0  # matching starts at 1

    for step in range(next_step, cfg.matching_iterations + 1):
        is_final = step == cfg.matching_iterations

        # TODO
        # prop = 1.0 if is_final else cfg.intermediate_matching_subsampling

        logger.dartsortdebug(f"-- Matching {step}")
        sorting = match(
            output_dir=store_dir,
            recording=recording,
            sorting=sorting,
            motion_est=motion_est,
            template_cfg=cfg.template_cfg,
            waveform_cfg=cfg.waveform_cfg,
            featurization_cfg=cfg.featurization_cfg,
            matching_cfg=cfg.matching_cfg,
            overwrite=overwrite or cfg.overwrite_matching,
            computation_cfg=cfg.computation_cfg,
            hdf5_filename=f"matching{step}.h5",
            model_subdir=f"matching{step}_models",
        )
        logger.info(f"Matching step {step}: {sorting}")
        ds_save_features(cfg, sorting, output_dir, work_dir, is_final)

        if cfg.final_refinement or not is_final:
            step_clustering_cfg = step_features_cfg = None
            if cfg.recluster_after_first_matching:
                step_clustering_cfg = cfg.clustering_cfg
                step_features_cfg = cfg.clustering_features_cfg
            sorting = cluster(
                recording,
                sorting,
                motion_est=motion_est,
                refinement_cfg=cfg.refinement_cfg,
                clustering_cfg=step_clustering_cfg,
                clustering_features_cfg=step_features_cfg,
                _save_cfg=cfg,
                _save_dir=output_dir,
                _save_initial_name=f"recluster{step}",
                _save_refined_name_fmt=f"refined{step}{{stepname}}",
            )

    if work_dir is not None:
        final_h5_path = output_dir / sorting.parent_h5_path.name
        assert final_h5_path.exists()
        sorting = replace(sorting, parent_h5_path=final_h5_path)
    ds_handle_delete_intermediate_features(cfg, sorting, output_dir, work_dir)

    sorting.save(output_dir / "dartsort_sorting.npz")
    ret["sorting"] = sorting
    return ret


def initial_detection(
    output_dir: str | Path,
    recording,
    cfg: DARTsortInternalConfig,
    motion_est=None,
    overwrite=False,
    show_progress=True,
):
    if cfg.detection_type == "subtract":
        assert isinstance(cfg.initial_detection_cfg, SubtractionConfig)
        return subtract(
            output_dir=output_dir,
            recording=recording,
            waveform_cfg=cfg.waveform_cfg,
            featurization_cfg=cfg.featurization_cfg,
            subtraction_cfg=cfg.initial_detection_cfg,
            computation_cfg=cfg.computation_cfg,
            overwrite=overwrite,
            show_progress=show_progress,
        )
    elif cfg.detection_type == "threshold":
        assert isinstance(cfg.initial_detection_cfg, ThresholdingConfig)
        return threshold(
            output_dir=output_dir,
            recording=recording,
            waveform_cfg=cfg.waveform_cfg,
            thresholding_cfg=cfg.initial_detection_cfg,
            featurization_cfg=cfg.featurization_cfg,
            overwrite=overwrite,
            show_progress=show_progress,
            computation_cfg=cfg.computation_cfg,
        )
    elif cfg.detection_type == "match":
        assert isinstance(cfg.initial_detection_cfg, MatchingConfig)
        return match(
            output_dir=output_dir,
            recording=recording,
            waveform_cfg=cfg.waveform_cfg,
            template_cfg=cfg.template_cfg,
            featurization_cfg=cfg.featurization_cfg,
            matching_cfg=cfg.initial_detection_cfg,
            motion_est=motion_est,
            overwrite=overwrite,
            show_progress=show_progress,
            computation_cfg=cfg.computation_cfg,
        )
    elif cfg.detection_type == "universal":
        assert isinstance(cfg.initial_detection_cfg, UniversalMatchingConfig)
        return universal_match(
            output_dir=output_dir,
            recording=recording,
            universal_cfg=cfg.initial_detection_cfg,
            featurization_cfg=cfg.featurization_cfg,
            overwrite=overwrite,
            show_progress=show_progress,
            computation_cfg=cfg.computation_cfg,
        )
    else:
        raise ValueError(f"Unknown detection_type {cfg.detection_type}.")


def subtract(
    output_dir: str | Path,
    recording: BaseRecording,
    waveform_cfg=default_waveform_cfg,
    featurization_cfg=default_featurization_cfg,
    subtraction_cfg=default_subtraction_cfg,
    computation_cfg: ComputationConfig | None = None,
    chunk_starts_samples=None,
    overwrite=False,
    residual_filename: str | None = None,
    show_progress=True,
    hdf5_filename="subtraction.h5",
    model_subdir="subtraction_models",
) -> DARTsortSorting | None:
    output_dir = resolve_path(output_dir)
    check_recording(recording)
    subtraction_peeler = SubtractionPeeler.from_config(
        recording,
        waveform_cfg=waveform_cfg,
        subtraction_cfg=subtraction_cfg,
        featurization_cfg=featurization_cfg,
    )
    detections = run_peeler(
        subtraction_peeler,
        output_dir,
        hdf5_filename,
        model_subdir=model_subdir,
        featurization_cfg=featurization_cfg,
        chunk_starts_samples=chunk_starts_samples,
        overwrite=overwrite,
        computation_cfg=computation_cfg,
        residual_filename=residual_filename,
        show_progress=show_progress,
        fit_only=subtraction_cfg.fit_only,
    )
    return detections


def match(
    output_dir: str | Path,
    recording: BaseRecording,
    sorting: DARTsortSorting | None = None,
    motion_est=None,
    waveform_cfg=default_waveform_cfg,
    template_cfg=default_template_cfg,
    featurization_cfg=default_featurization_cfg,
    matching_cfg=default_matching_cfg,
    chunk_starts_samples=None,
    overwrite=False,
    residual_filename: str | None = None,
    show_progress=True,
    hdf5_filename="matching0.h5",
    model_subdir="matching0_models",
    template_data: TemplateData | None = None,
    template_npz_filename="template_data.npz",
    computation_cfg: ComputationConfig | None = None,
    template_denoising_tsvd=None,
) -> DARTsortSorting:
    output_dir = resolve_path(output_dir)
    model_dir = output_dir / model_subdir

    # compute templates
    if template_data is None and not matching_cfg.precomputed_templates_npz:
        sorting, template_data = postprocess(
            recording,
            sorting,
            motion_est=motion_est,
            matching_cfg=matching_cfg,
            waveform_cfg=waveform_cfg,
            template_cfg=template_cfg,
            computation_cfg=computation_cfg,
            tsvd=template_denoising_tsvd,
            template_npz_path=model_dir / template_npz_filename,
        )

    # instantiate peeler
    matching_peeler = ObjectiveUpdateTemplateMatchingPeeler.from_config(
        recording,
        waveform_cfg,
        matching_cfg,
        featurization_cfg,
        template_data,
        motion_est=motion_est,
    )
    sorting = run_peeler(
        matching_peeler,
        output_dir,
        hdf5_filename,
        model_subdir,
        featurization_cfg,
        chunk_starts_samples=chunk_starts_samples,
        overwrite=overwrite,
        residual_filename=residual_filename,
        show_progress=show_progress,
        computation_cfg=computation_cfg,
    )
    assert sorting is not None
    return sorting


def grab(
    output_dir: str | Path,
    recording: BaseRecording,
    sorting: DARTsortSorting,
    waveform_cfg=default_waveform_cfg,
    featurization_cfg=default_featurization_cfg,
    chunk_starts_samples=None,
    overwrite=False,
    show_progress=True,
    hdf5_filename="grab.h5",
    model_subdir="grab_models",
    computation_cfg: ComputationConfig | None = None,
) -> DARTsortSorting:
    output_dir = resolve_path(output_dir)
    grabber = GrabAndFeaturize.from_config(
        sorting, recording, waveform_cfg, featurization_cfg
    )
    sorting = run_peeler(
        grabber,
        output_dir,
        hdf5_filename,
        model_subdir,
        featurization_cfg,
        chunk_starts_samples=chunk_starts_samples,
        overwrite=overwrite,
        show_progress=show_progress,
        computation_cfg=computation_cfg,
    )
    return sorting


def threshold(
    output_dir: str | Path,
    recording: BaseRecording,
    waveform_cfg=default_waveform_cfg,
    thresholding_cfg=default_thresholding_cfg,
    featurization_cfg=default_featurization_cfg,
    chunk_starts_samples=None,
    overwrite=False,
    show_progress=True,
    hdf5_filename="threshold.h5",
    model_subdir="threshold_models",
    computation_cfg: ComputationConfig | None = None,
) -> DARTsortSorting:
    output_dir = resolve_path(output_dir)
    thresholder = ThresholdAndFeaturize.from_config(
        recording, waveform_cfg, thresholding_cfg, featurization_cfg
    )
    sorting = run_peeler(
        thresholder,
        output_dir,
        hdf5_filename,
        model_subdir,
        featurization_cfg,
        chunk_starts_samples=chunk_starts_samples,
        overwrite=overwrite,
        show_progress=show_progress,
        computation_cfg=computation_cfg,
    )
    return sorting


def cluster(
    recording,
    sorting,
    motion_est=None,
    clustering_cfg: ClusteringConfig | None = default_clustering_cfg,
    clustering_features_cfg: (
        ClusteringFeaturesConfig | None
    ) = default_clustering_features_cfg,
    refinement_cfg: RefinementConfig | None = None,
    computation_cfg: ComputationConfig | None = None,
    *,
    _save_cfg: DARTsortInternalConfig | None = None,
    _save_initial_name="initial",
    _save_refined_name_fmt="refined0{stepname}",
    _save_dir=None,
):
    features = get_clustering_features(
        recording,
        sorting,
        motion_est=motion_est,
        clustering_features_cfg=clustering_features_cfg,
    )
    clusterer = get_clusterer(
        clustering_cfg=clustering_cfg,
        refinement_cfg=refinement_cfg,
        computation_cfg=computation_cfg,
        save_cfg=_save_cfg,
        save_labels_dir=_save_dir,
        initial_name=_save_initial_name,
        refine_labels_fmt=_save_refined_name_fmt,
    )
    return clusterer.cluster(
        recording=recording, sorting=sorting, features=features, motion_est=motion_est
    )


def universal_match(
    output_dir: str | Path,
    recording: BaseRecording,
    universal_cfg=default_universal_cfg,
    featurization_cfg=default_featurization_cfg,
    chunk_starts_samples=None,
    overwrite=False,
    show_progress=True,
    hdf5_filename="universal.h5",
    model_subdir="universal_models",
    computation_cfg: ComputationConfig | None = None,
) -> DARTsortSorting:
    output_dir = resolve_path(output_dir)
    universal_matcher = UniversalTemplatesMatchingPeeler.from_config(
        recording, universal_cfg, featurization_cfg
    )
    sorting = run_peeler(
        universal_matcher,
        output_dir,
        hdf5_filename,
        model_subdir,
        featurization_cfg,
        chunk_starts_samples=chunk_starts_samples,
        overwrite=overwrite,
        show_progress=show_progress,
        computation_cfg=computation_cfg,
    )
    assert sorting is not None
    return sorting


def match_chunked(
    recording,
    sorting,
    output_dir=None,
    motion_est=None,
    waveform_cfg=default_waveform_cfg,
    template_cfg=default_template_cfg,
    featurization_cfg=default_featurization_cfg,
    matching_cfg=default_matching_cfg,
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
    chunk_samples = recording.sampling_frequency * template_cfg.chunk_size_s
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
            n_min_spikes=template_cfg.spikes_per_unit,
            latest_time_sample=chunk_time_range[1] * recording.sampling_frequency,
        )
        chunk_starts_samples = recording._recording_segments[0].time_to_sample_index(
            chunk_time_range
        )
        chunk_starts_samples = chunk_starts_samples.astype(int)
        chunk_starts_samples = np.arange(
            *chunk_starts_samples, matching_cfg.chunk_length_samples
        )

        chunk_sorting, chunk_h5 = match(
            recording,
            sorting=sorting_chunk,
            output_dir=output_dir,
            motion_est=motion_est,
            waveform_cfg=default_waveform_cfg,
            template_cfg=default_template_cfg,
            featurization_cfg=default_featurization_cfg,
            matching_cfg=default_matching_cfg,
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
