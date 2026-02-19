import gc
import traceback
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from spikeinterface.core import BaseRecording

from .clustering import SimpleMatrixFeatures, get_clusterer, get_clustering_features
from .config import DARTsortUserConfig, DeveloperConfig
from .peel import (
    GrabAndFeaturize,
    ObjectiveUpdateTemplateMatchingPeeler,
    SubtractionPeeler,
    ThresholdAndFeaturize,
    UniversalTemplatesMatchingPeeler,
)
from .templates import TemplateData, estimate_template_library
from .util.data_util import DARTsortSorting, check_recording
from .util.internal_config import (
    ClusteringConfig,
    ClusteringFeaturesConfig,
    ComputationConfig,
    DARTsortInternalConfig,
    MatchingConfig,
    RefinementConfig,
    SubtractionConfig,
    ThresholdingConfig,
    UniversalMatchingConfig,
    default_clustering_cfg,
    default_clustering_features_cfg,
    default_dartsort_cfg,
    default_featurization_cfg,
    default_matching_cfg,
    default_peeling_fit_sampling_cfg,
    default_subtraction_cfg,
    default_template_cfg,
    default_thresholding_cfg,
    default_universal_cfg,
    default_waveform_cfg,
    to_internal_config,
)
from .util.logging_util import get_logger
from .util.main_util import (
    ds_all_to_workdir,
    ds_dump_config,
    ds_fast_forward,
    ds_handle_delete_intermediate_features,
    ds_handle_link_from,
    ds_save_features,
    ds_save_intermediate_labels,
    ds_save_motion_est,
)
from .util.peel_util import run_peeler
from .util.py_util import dartcopytree, resolve_path
from .util.registration_util import estimate_motion

logger = get_logger(__name__)


def dartsort(
    recording: BaseRecording,
    output_dir: str | Path,
    cfg: (
        DARTsortUserConfig | str | Path | DeveloperConfig | DARTsortInternalConfig
    ) = default_dartsort_cfg,
    motion_est=None,
    overwrite=False,
):
    """dartsort

    This function runs a spike sorter called dartsort.

    Arguments
    ---------
    recording : BaseRecording
        A spikeinterface.BaseRecording object
    output_dir: str | Path
        Folder where outputs are stored
    cfg: DARTsortUserConfig | DARTsortInternalConfig | str | Path
        Your settings. Either create a DARTsortUserConfig directly in code, or
        you can pass a string or Path pointing to a .toml file here.
    motion_est: optional dredge.MotionEstimate
        This is meant to allow users to pass their own external motion estimate.
        To do: support spikeinterface Motion objects here.
    overwrite : bool, default=False
        Ignore and overwrite stored results, if any. Otherwise, dartsort will
        try to resume from the last step that ran, or if it had finished then
        it will do nothing.

    Returns
    -------
    Dictionary of sorting results, with the following keys:
     - "sorting": DARTsortSorting
        This has a sorting.to_numpy_sorting() method for those who want to
        export back to spikeinterface. Which would be everyone? :)
        Alternatively, you could visualize your results using the functions
        in the `import dartsort.vis as dartvis` library.
     - "motion_est": dredge.MotionEstimate

    TODO: add the key "motion" with a spikeinterface Motion object.
    """
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # convert cfg to internal format and store it for posterity
    cfg = to_internal_config(cfg)
    ds_dump_config(cfg, output_dir)

    # in benchmarking, it can be useful to resume from initial detection
    # and/or clustering results stored in elsewhere to avoid rerunning
    ds_handle_link_from(cfg, output_dir)

    if cfg.work_in_tmpdir:
        with TemporaryDirectory(prefix="dartsort", dir=cfg.tmpdir_parent) as work_dir:
            # copy files and possibly recording to temporary directory
            work_dir = resolve_path(work_dir)
            logger.dartsortdebug(f"Working in {work_dir}, outputs to {output_dir}.")
            recording, work_dir = ds_all_to_workdir(
                internal_cfg=cfg,
                output_dir=output_dir,
                work_dir=work_dir,
                recording=recording,
                overwrite=overwrite,
            )
            assert work_dir is not None

            # run the sorter, with extra error handlers for grabbing stuff from
            # the temporary directory if user asked for that.
            try:
                return _dartsort_impl(
                    recording=recording,
                    output_dir=output_dir,
                    cfg=cfg,
                    motion_est=motion_est,
                    work_dir=work_dir,
                    overwrite=overwrite,
                )
            except Exception as e:
                traceback_path = output_dir / "traceback.txt"
                error_data_path = output_dir / "error_state"
                with open(traceback_path, "w") as f:
                    traceback.print_exception(e, file=f)
                logger.exception(e)
                if cfg.save_everything_on_error:
                    logger.critical(
                        f"Hit an error. Copying outputs to {error_data_path} "
                        f"and writing traceback to {traceback_path}."
                    )
                    dartcopytree(cfg, work_dir, error_data_path)
                else:
                    logger.critical(
                        f"Hit an error. Writing traceback to {traceback_path}."
                        " work_in_tmpdir was true, so the files won't be kept."
                        " Set save_everything_on_error to keep them."
                    )
                raise

    # run the sorter regular with no tempdir, log exception to a
    # traceback file in case of a crash for debugging
    try:
        return _dartsort_impl(
            recording=recording,
            output_dir=output_dir,
            cfg=cfg,
            motion_est=motion_est,
            work_dir=None,
            overwrite=overwrite,
        )
    except Exception as e:
        traceback_path = output_dir / "traceback.txt"
        with open(traceback_path, "w") as f:
            traceback.print_exception(e, file=f)
        logger.exception(e)
        logger.critical(f"Hit an error. Wrote traceback to {traceback_path}.")
        raise


def _dartsort_impl(
    *,
    recording: BaseRecording,
    output_dir: Path,
    cfg: DARTsortInternalConfig = default_dartsort_cfg,
    motion_est=None,
    work_dir=None,
    overwrite=False,
):
    """Internal helper function which implements dartsort's main logic."""
    ret = {}

    # are we storing files in the work dir or the output dir?
    store_dir = output_dir if work_dir is None else work_dir

    # if there are previous results stored, resume where they leave off
    # TODO uhh. overwrite, right?
    next_step, sorting, _motion_est = ds_fast_forward(store_dir, cfg)
    if motion_est is None:
        motion_est = _motion_est
    ret["motion_est"] = motion_est

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
        r_cfgs = [
            cfg.pre_refinement_cfg,
            cfg.initial_refinement_cfg,
            cfg.post_refinement_cfg,
        ]
        sorting = cluster(
            recording,
            sorting,
            motion_est=motion_est,
            refinement_cfgs=r_cfgs,
            clustering_cfg=cfg.clustering_cfg,
            clustering_features_cfg=cfg.clustering_features_cfg,
            _save_cfg=cfg,
            _save_dir=output_dir,
        )
        logger.info(f"First clustering: {sorting}")
        ds_save_intermediate_labels(
            step_name="refined0",
            step_sorting=sorting,
            output_dir=output_dir,
            cfg=cfg,
            work_dir=work_dir,
        )

        # be sure to start matching at step 1
        next_step += 1

    assert sorting is not None
    assert (motion_est is not None) == cfg.motion_estimation_cfg.do_motion_estimation
    assert next_step > 0  # matching starts at 1

    for step in range(next_step, cfg.matching_iterations + 1):
        is_final = step == cfg.matching_iterations

        if step == 0:
            assert False
        elif step == 1:
            previous_detection_cfg = cfg.initial_detection_cfg
        else:
            previous_detection_cfg = cfg.matching_cfg

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
            overwrite=overwrite,
            computation_cfg=cfg.computation_cfg,
            hdf5_filename=f"matching{step}.h5",
            model_subdir=f"matching{step}_models",
            previous_detection_cfg=previous_detection_cfg,
            prev_step_name=f"refined{step - 1}",
            save_cfg=cfg,
        )
        logger.info(f"Matching step {step}: {sorting}")
        ds_save_features(cfg, sorting, output_dir, work_dir, is_final)

        if is_final and not cfg.final_refinement:
            break

        if cfg.recluster_after_first_matching:
            step_clustering_cfg = cfg.clustering_cfg
            step_features_cfg = cfg.clustering_features_cfg
        else:
            step_clustering_cfg = step_features_cfg = None

        r_cfgs = [
            cfg.pre_refinement_cfg,
            cfg.refinement_cfg,
            cfg.post_refinement_cfg,
        ]

        sorting = cluster(
            recording,
            sorting,
            motion_est=motion_est,
            refinement_cfgs=r_cfgs,
            clustering_cfg=step_clustering_cfg,
            clustering_features_cfg=step_features_cfg,
            _save_cfg=cfg,
            _save_dir=output_dir,
            _save_initial_name=f"recluster{step}",
            _save_refined_name_fmt=f"refined{step}{{stepname}}",
        )
        ds_save_intermediate_labels(
            step_name=f"refined{step}",
            step_sorting=sorting,
            output_dir=output_dir,
            cfg=cfg,
            work_dir=work_dir,
        )

    # finally handle scratch directory and delete intermediate files if requested
    if work_dir is not None:
        orig_h5_path = resolve_path(sorting.parent_h5_path, strict=True)
        final_h5_path = output_dir / orig_h5_path.name
        assert final_h5_path.exists()
        sorting.parent_h5_path = final_h5_path
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
            sampling_cfg=cfg.peeler_sampling_cfg,
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
            sampling_cfg=cfg.peeler_sampling_cfg,
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
            sampling_cfg=cfg.peeler_sampling_cfg,
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
            sampling_cfg=cfg.peeler_sampling_cfg,
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
    sampling_cfg=default_peeling_fit_sampling_cfg,
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
        recording=recording,
        sampling_cfg=sampling_cfg,
        waveform_cfg=waveform_cfg,
        subtraction_cfg=subtraction_cfg,
        featurization_cfg=featurization_cfg,
    )
    detections = run_peeler(
        subtraction_peeler,
        output_directory=output_dir,
        hdf5_filename=hdf5_filename,
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
    sampling_cfg=default_peeling_fit_sampling_cfg,
    previous_detection_cfg: Any | None = None,
    prev_step_name: str | None = None,
    save_cfg: DARTsortInternalConfig | None = None,
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

    if template_data is None and not matching_cfg.precomputed_templates_npz:
        assert sorting is not None
        sorting, template_data = estimate_template_library(
            recording=recording,
            sorting=sorting,
            motion_est=motion_est,
            min_template_ptp=matching_cfg.min_template_ptp,
            min_template_snr=matching_cfg.min_template_snr,
            min_template_count=matching_cfg.min_template_count,
            max_cc_flag_rate=matching_cfg.max_cc_flag_rate,
            cc_flag_entropy_cutoff=matching_cfg.cc_flag_entropy_cutoff,
            depth_order=matching_cfg.depth_order,
            waveform_cfg=waveform_cfg,
            template_cfg=template_cfg,
            realign_cfg=matching_cfg.template_realignment_cfg,
            template_merge_cfg=matching_cfg.template_merge_cfg,
            computation_cfg=computation_cfg,
            detection_cfg=previous_detection_cfg,
            tsvd=template_denoising_tsvd,
            template_npz_path=model_dir / template_npz_filename,
        )
        if prev_step_name is not None:
            ds_save_intermediate_labels(
                step_name=f"{prev_step_name}_9_prematch",
                step_sorting=sorting,
                output_dir=output_dir,
                cfg=save_cfg,
            )

    matching_peeler = ObjectiveUpdateTemplateMatchingPeeler.from_config(
        recording=recording,
        waveform_cfg=waveform_cfg,
        matching_cfg=matching_cfg,
        sampling_cfg=sampling_cfg,
        featurization_cfg=featurization_cfg,
        template_data=template_data,
        motion_est=motion_est,
        parent_sorting_hdf5_path=getattr(sorting, "parent_h5_path", None),
    )
    sorting = run_peeler(
        matching_peeler,
        output_directory=output_dir,
        hdf5_filename=hdf5_filename,
        model_subdir=model_subdir,
        featurization_cfg=featurization_cfg,
        chunk_starts_samples=chunk_starts_samples,
        overwrite=overwrite,
        residual_filename=residual_filename,
        show_progress=show_progress,
        computation_cfg=computation_cfg,
    )
    return sorting


def grab(
    output_dir: str | Path,
    recording: BaseRecording,
    sorting: DARTsortSorting,
    waveform_cfg=default_waveform_cfg,
    featurization_cfg=default_featurization_cfg,
    sampling_cfg=default_peeling_fit_sampling_cfg,
    chunk_starts_samples=None,
    overwrite=False,
    show_progress=True,
    hdf5_filename="grab.h5",
    model_subdir="grab_models",
    computation_cfg: ComputationConfig | None = None,
) -> DARTsortSorting:
    output_dir = resolve_path(output_dir)
    grabber = GrabAndFeaturize.from_config(
        sorting=sorting,
        recording=recording,
        waveform_cfg=waveform_cfg,
        sampling_cfg=sampling_cfg,
        featurization_cfg=featurization_cfg,
    )
    sorting = run_peeler(
        grabber,
        output_directory=output_dir,
        hdf5_filename=hdf5_filename,
        model_subdir=model_subdir,
        featurization_cfg=featurization_cfg,
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
    sampling_cfg=default_peeling_fit_sampling_cfg,
    chunk_starts_samples=None,
    overwrite=False,
    show_progress=True,
    hdf5_filename="threshold.h5",
    model_subdir="threshold_models",
    computation_cfg: ComputationConfig | None = None,
) -> DARTsortSorting:
    output_dir = resolve_path(output_dir)
    thresholder = ThresholdAndFeaturize.from_config(
        recording=recording,
        waveform_cfg=waveform_cfg,
        thresholding_cfg=thresholding_cfg,
        featurization_cfg=featurization_cfg,
        sampling_cfg=sampling_cfg,
    )
    sorting = run_peeler(
        thresholder,
        output_directory=output_dir,
        hdf5_filename=hdf5_filename,
        model_subdir=model_subdir,
        featurization_cfg=featurization_cfg,
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
    refinement_cfgs: list[RefinementConfig | None] | None = None,
    computation_cfg: ComputationConfig | None = None,
    features: SimpleMatrixFeatures | None = None,
    *,
    _save_cfg: DARTsortInternalConfig | None = None,
    _save_initial_name="initial",
    _save_refined_name_fmt="refined0{stepname}",
    _save_dir=None,
):
    if features is None:
        features = get_clustering_features(
            recording,
            sorting,
            motion_est=motion_est,
            clustering_features_cfg=clustering_features_cfg,
        )
    assert features is not None
    clusterer = get_clusterer(
        clustering_cfg=clustering_cfg,
        refinement_cfgs=refinement_cfgs,
        computation_cfg=computation_cfg,
        save_cfg=_save_cfg,
        save_labels_dir=_save_dir,
        initial_name=_save_initial_name,
        refine_labels_fmt=_save_refined_name_fmt,
    )
    result = clusterer.cluster(
        recording=recording, sorting=sorting, features=features, motion_est=motion_est
    )

    del features, clusterer
    gc.collect()
    torch.cuda.empty_cache()

    return result


def universal_match(
    output_dir: str | Path,
    recording: BaseRecording,
    universal_cfg=default_universal_cfg,
    featurization_cfg=default_featurization_cfg,
    sampling_cfg=default_peeling_fit_sampling_cfg,
    chunk_starts_samples=None,
    overwrite=False,
    show_progress=True,
    hdf5_filename="universal.h5",
    model_subdir="universal_models",
    computation_cfg: ComputationConfig | None = None,
) -> DARTsortSorting:
    output_dir = resolve_path(output_dir)
    universal_matcher = UniversalTemplatesMatchingPeeler.from_config(
        recording=recording,
        universal_cfg=universal_cfg,
        waveform_cfg=universal_cfg.waveform_cfg,  # TODO...
        featurization_cfg=featurization_cfg,
        sampling_cfg=sampling_cfg,
    )
    sorting = run_peeler(
        universal_matcher,
        output_directory=output_dir,
        hdf5_filename=hdf5_filename,
        model_subdir=model_subdir,
        featurization_cfg=featurization_cfg,
        chunk_starts_samples=chunk_starts_samples,
        overwrite=overwrite,
        show_progress=show_progress,
        computation_cfg=computation_cfg,
    )
    return sorting
