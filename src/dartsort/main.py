"""High-level spike sorting toolbox functions."""

import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence, TypedDict

from dredge.motion_util import MotionEstimate
from spikeinterface.core import BaseRecording, Motion
from torch import Tensor

from .clustering import SimpleMatrixFeatures, StableWaveformFeatures, get_clusterer
from .config import DARTsortUserConfig, DeveloperConfig
from .peel import (
    GrabAndFeaturize,
    ObjectiveUpdateTemplateMatchingPeeler,
    SubtractionPeeler,
    Threshold,
)
from .templates import TemplateData, estimate_template_library
from .transform import WaveformPipeline
from .util.data_util import DARTsortSorting, check_recording
from .util.internal_config import (
    ClusteringConfig,
    ClusteringFeaturesConfig,
    ComputationConfig,
    DARTsortInternalConfig,
    FeaturizationConfig,
    FitSamplingConfig,
    MatchingConfig,
    RefinementConfig,
    SubtractionConfig,
    ThresholdingConfig,
    WaveformConfig,
    default_clustering_cfg,
    default_clustering_features_cfg,
    default_dartsort_cfg,
    default_featurization_cfg,
    default_matching_cfg,
    default_peeling_fit_sampling_cfg,
    default_subtraction_cfg,
    default_template_cfg,
    default_thresholding_cfg,
    default_waveform_cfg,
    to_internal_config,
)
from .util.job_util import ensure_computation_config
from .util.logging_util import get_logger
from .util.main_util import (
    _matching_step_cfgs,
    ds_all_to_workdir,
    ds_dump_config,
    ds_fast_forward,
    ds_handle_delete_intermediate_features,
    ds_handle_link_from,
    ds_save_features,
    ds_save_intermediate_labels,
    ds_save_motion,
    ds_save_timing,
    motion_needs_peaks,
)
from .util.motion import MotionInfo, get_motion_info
from .util.noise_util import Whitener
from .util.peel_util import run_peeler
from .util.preprocess_util import preprocess
from .util.py_util import dartcopytree, ensure_path, timer
from .util.torch_util import cleanup_and_log_gpu_usage

logger = get_logger(__name__)


class DARTsortResult(TypedDict):
    sorting: DARTsortSorting
    """Output spike trains."""
    motion: MotionInfo
    """Esimated motion"""


def dartsort(
    recording: BaseRecording,
    output_dir: str | Path,
    cfg: (
        DARTsortUserConfig | str | Path | DeveloperConfig | DARTsortInternalConfig
    ) = default_dartsort_cfg,
    motion: MotionInfo | None = None,
    si_motion: Motion | None = None,
    dredge_motion_est: MotionEstimate | None = None,
    overwrite=False,
):
    """This function runs a spike sorter called *dartsort*.

    Parameters
    ---------
    recording : BaseRecording
        A SpikeInterface `BaseRecording`
    output_dir : str or Path
        Folder where outputs are stored. See the `work_in_tmpdir` and `tmpdir_parent`
        configuration options to store intermediate data in a scratch folder and
        then only save the final outputs here.
    cfg : DARTsortUserConfig or DARTsortInternalConfig or str or Path
        Your settings. Either create a `DARTsortUserConfig` directly in code, or
        you can pass a string or Path pointing to a .toml file here.
    si_motion : spikeinterface.core.Motion, optional
        Allows users to pass their own external motion estimate. If this is given,
        the do_motion_estimation configuration flag is ignored and this object is
        used.
    dredge_motion_est : dredge.MotionEstimate, optional
        As in `si_motion`.
    overwrite : bool
        Ignore and overwrite stored results, if any. Otherwise, dartsort will
        try to resume from the last step that ran, or if it had finished then
        it will do nothing.

    Returns
    -------
    results : DARTsortResult
        Dictionary of sorting results, with keys:

          - "sorting": `DARTsortSorting`
          - "motion": MotionInfo
    """
    output_dir = ensure_path(output_dir, mkdir=True)

    # convert cfg to internal format and store it for posterity
    cfg = to_internal_config(cfg, recording.get_num_channels())
    ds_dump_config(cfg, output_dir)

    # in benchmarking, it can be useful to resume from initial detection
    # and/or clustering results stored in elsewhere to avoid rerunning
    ds_handle_link_from(cfg, output_dir)

    # preprocess
    recording = preprocess(recording, cfg.preprocessing, cfg.preprocessing_dtype)

    if cfg.work_in_tmpdir:
        with TemporaryDirectory(prefix="dartsort", dir=cfg.tmpdir_parent) as work_dir:
            # copy files and possibly recording to temporary directory
            work_dir = ensure_path(work_dir)
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
                    motion=motion,
                    si_motion=si_motion,
                    dredge_motion_est=dredge_motion_est,
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
            motion=motion,
            si_motion=si_motion,
            dredge_motion_est=dredge_motion_est,
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
    motion: MotionInfo | None = None,
    si_motion: Motion | None = None,
    dredge_motion_est: MotionEstimate | None = None,
    work_dir: Path | None = None,
    overwrite=False,
):
    """Internal helper function which implements dartsort's main logic."""
    ret: dict[str, Any] = {"timing": {}}

    total_timer = timer("total", ret["timing"])
    total_timer.start()

    # are we storing files in the work dir or the output dir?
    store_dir = output_dir if work_dir is None else work_dir

    # if there are previous results stored, resume where they leave off
    # TODO uhh. overwrite, right?
    next_step, sorting, _motion = ds_fast_forward(store_dir, cfg)
    if motion is not None:
        pass
    elif (si_motion is None) and (dredge_motion_est is None):
        motion = _motion
    else:
        motion = MotionInfo.from_motion_est(
            geom=recording.get_channel_locations(),
            dredge_motion_est=dredge_motion_est,
            si_motion=si_motion,
        )
    if motion is None and next_step > 0:
        assert sorting is not None
        logger.dartsortdebug("-- Estimate motion")
        motion = get_motion_info(
            output_directory=store_dir,
            recording=recording,
            sorting=sorting,
            detect_new_peaks=motion_needs_peaks(cfg, recording, sorting),
            motion_cfg=cfg.motion_estimation_cfg,
            computation_cfg=cfg.computation_cfg,
            sampling_cfg=cfg.peeler_sampling_cfg,
            waveform_cfg=cfg.waveform_cfg,
            overwrite=overwrite,
            _save_cfg=cfg,
            _save_dir=output_dir,
        )
    ret["motion"] = motion

    is_subsampling = cfg.subsampling_spikes_per_channel is not None
    is_subsampling = is_subsampling and cfg.subsampling_presence != 1.0

    if next_step == 0:
        # first step: initial detection and motion estimation
        with timer("initial_detection", ret["timing"]):
            sorting = initial_detection(
                output_dir=store_dir,
                recording=recording,
                cfg=cfg,
                overwrite=overwrite,
                motion=motion,
            )
        assert sorting is not None
        logger.info(f"Initial detection: {sorting}")
        is_final = cfg.detect_only or cfg.dredge_only or not cfg.matching_iterations
        ds_save_features(cfg, sorting, output_dir, work_dir, is_final=is_final)

        if cfg.detect_only:
            ret["sorting"] = sorting
            return ret

        with timer("motion", ret["timing"]):
            if motion is None:
                logger.dartsortdebug("-- Estimate motion")
                motion = get_motion_info(
                    output_directory=store_dir,
                    recording=recording,
                    sorting=sorting,
                    detect_new_peaks=motion_needs_peaks(cfg, recording, sorting),
                    motion_cfg=cfg.motion_estimation_cfg,
                    computation_cfg=cfg.computation_cfg,
                    sampling_cfg=cfg.peeler_sampling_cfg,
                    waveform_cfg=cfg.waveform_cfg,
                    overwrite=overwrite,
                    _save_cfg=cfg,
                    _save_dir=output_dir,
                )
            ret["motion"] = motion
            ds_save_motion(motion, output_dir, work_dir, overwrite)

        if cfg.dredge_only:
            ret["sorting"] = sorting
            return ret

        # clustering: initialization and first refinement
        r_cfgs = [
            cfg.pre_refinement_cfg,
            cfg.initial_refinement_cfg,
            cfg.post_refinement_cfg,
        ]
        with timer("cluster0", ret["timing"]):
            sorting = cluster(
                recording,
                sorting,
                motion=motion,
                refinement_cfgs=r_cfgs,
                computation_cfg=cfg.computation_cfg,
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
    assert motion is not None
    assert next_step > 0  # matching starts at 1

    for step in range(next_step, cfg.matching_iterations + 1):
        is_final = step == cfg.matching_iterations

        if step == 0:
            assert False
        elif step == 1:
            previous_detection_cfg = cfg.initial_detection_cfg
        else:
            previous_detection_cfg = cfg.matching_cfg

        if is_final or cfg.subsampling_spikes_per_channel is None:
            _nspk = None
        else:
            _nspk = cfg.subsampling_spikes_per_channel * motion.geom.shape[0]
        _pres = 1.0 if is_final else cfg.subsampling_presence
        step_clus_cfg, step_clfeat_cfg, step_ref_cfgs, step_feat_cfg, samp_cfg = (
            _matching_step_cfgs(is_final, is_subsampling, cfg)
        )

        logger.dartsortdebug(f"-- Matching {step}")
        with timer(f"matching{step}", ret["timing"]):
            sorting = match(
                output_dir=store_dir,
                recording=recording,
                sorting=sorting,
                motion=motion,
                sampling_cfg=samp_cfg,
                template_cfg=cfg.template_cfg,
                waveform_cfg=cfg.waveform_cfg,
                featurization_cfg=step_feat_cfg,
                matching_cfg=cfg.matching_cfg,
                overwrite=overwrite,
                computation_cfg=cfg.computation_cfg,
                stop_after_n_spikes=_nspk,
                ensure_coverage=_pres,
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

        with timer(f"cluster{step}", ret["timing"]):
            if step_clus_cfg or step_ref_cfgs is not None and len(step_ref_cfgs):
                sorting = cluster(
                    recording=recording,
                    sorting=sorting,
                    motion=motion,
                    computation_cfg=cfg.computation_cfg,
                    refinement_cfgs=step_ref_cfgs,
                    clustering_cfg=step_clus_cfg,
                    clustering_features_cfg=step_clfeat_cfg,
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
        orig_h5_path = ensure_path(sorting.parent_h5_path, strict=True)
        final_h5_path = output_dir / orig_h5_path.name
        assert final_h5_path.exists()
        sorting.parent_h5_path = final_h5_path
    ds_handle_delete_intermediate_features(cfg, sorting, output_dir, work_dir)

    sorting.save(output_dir / "dartsort_sorting.npz")
    ret["sorting"] = sorting

    total_timer.stop()
    logger.dartsortdebug(f"Timing: {ret['timing']}")
    ds_save_timing(ret["timing"], output_dir)

    return ret


def initial_detection(
    output_dir: str | Path,
    recording: BaseRecording,
    cfg: DARTsortInternalConfig,
    motion: MotionInfo | None = None,
    overwrite=False,
    show_progress=True,
) -> DARTsortSorting:
    """Initial spike detection

    Runs the detection method specified by cfg.detection_type

    Used by dartsort; users probably want to just run subtract(), match(),
    or threshold() directly.

    Parameters
    ----------
    output_dir : str | Path
    recording : BaseRecording
    cfg : DARTsortInternalConfig
    motion : MotionInfo | None, optional
    overwrite : bool, optional
    show_progress : bool, optional

    Returns
    -------
    DARTsortSorting
    """
    if cfg.subsampling_spikes_per_channel is None:
        _nspk = None
    else:
        _nspk = cfg.subsampling_spikes_per_channel * recording.get_num_channels()
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
            stop_after_n_spikes=_nspk,
            ensure_coverage=cfg.subsampling_presence,
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
            stop_after_n_spikes=_nspk,
            ensure_coverage=cfg.subsampling_presence,
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
            motion=motion,
            stop_after_n_spikes=_nspk,
            ensure_coverage=cfg.subsampling_presence,
            overwrite=overwrite,
            show_progress=show_progress,
            computation_cfg=cfg.computation_cfg,
        )
    else:
        raise ValueError(f"Unknown detection_type {cfg.detection_type}.")


def subtract(
    output_dir: str | Path,
    recording: BaseRecording,
    waveform_cfg: WaveformConfig = default_waveform_cfg,
    featurization_cfg: FeaturizationConfig = default_featurization_cfg,
    subtraction_cfg=default_subtraction_cfg,
    sampling_cfg: FitSamplingConfig = default_peeling_fit_sampling_cfg,
    computation_cfg: ComputationConfig | None = None,
    chunk_starts_samples=None,
    stop_after_n_spikes: int | None = None,
    ensure_coverage: float | None = None,
    overwrite=False,
    residual_filename: str | None = None,
    shuffle: bool = False,
    show_progress=True,
    hdf5_filename="subtraction.h5",
    model_subdir="subtraction_models",
) -> DARTsortSorting:
    output_dir = ensure_path(output_dir)
    computation_cfg = ensure_computation_config(computation_cfg)
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
        stop_after_n_spikes=stop_after_n_spikes,
        ensure_coverage=ensure_coverage,
        shuffle=shuffle,
    )

    del subtraction_peeler
    cleanup_and_log_gpu_usage(computation_cfg, f"Post subtract ({hdf5_filename}):")

    return detections


def match(
    output_dir: str | Path,
    recording: BaseRecording,
    sorting: DARTsortSorting | None = None,
    motion: MotionInfo | None = None,
    waveform_cfg: WaveformConfig = default_waveform_cfg,
    template_cfg=default_template_cfg,
    featurization_cfg: FeaturizationConfig = default_featurization_cfg,
    matching_cfg=default_matching_cfg,
    sampling_cfg: FitSamplingConfig = default_peeling_fit_sampling_cfg,
    previous_detection_cfg: Any | None = None,
    prev_step_name: str | None = None,
    save_cfg: DARTsortInternalConfig | None = None,
    chunk_starts_samples=None,
    stop_after_n_spikes: int | None = None,
    ensure_coverage: float | None = None,
    overwrite=False,
    residual_filename: str | None = None,
    skip_resid_snips=False,
    show_progress=True,
    hdf5_filename="matching0.h5",
    model_subdir="matching0_models",
    template_data: TemplateData | None = None,
    template_npz="template_data.npz",
    computation_cfg: ComputationConfig | None = None,
    template_denoising_tsvd=None,
    whitener: Whitener | None = None,
) -> DARTsortSorting:
    output_dir = ensure_path(output_dir)
    model_dir = output_dir / model_subdir
    computation_cfg = ensure_computation_config(computation_cfg)

    if template_data is None and not matching_cfg.precomputed_templates_npz:
        assert sorting is not None
        assert template_cfg.whitening == matching_cfg.whitening
        sorting, template_data = estimate_template_library(
            recording=recording,
            sorting=sorting,
            motion=motion,
            min_template_ptp=matching_cfg.min_template_ptp,
            always_keep_ptp=matching_cfg.always_keep_ptp,
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
            fit_featurization_tsvd=featurization_cfg.tpca_from_templates,
            featurization_cfg=featurization_cfg,
            tsvd=template_denoising_tsvd,
            whitener=whitener,
            template_npz_path=model_dir / template_npz,
        )
        if prev_step_name is not None:
            assert sorting is not None
            ds_save_intermediate_labels(
                step_name=f"{prev_step_name}_9_prematch",
                step_sorting=sorting,
                output_dir=output_dir,
                cfg=save_cfg,
            )
        cleanup_and_log_gpu_usage(
            computation_cfg, f"Post templates ({model_subdir}/{template_npz}):"
        )

    matching_peeler = ObjectiveUpdateTemplateMatchingPeeler.from_config(
        recording=recording,
        waveform_cfg=waveform_cfg,
        matching_cfg=matching_cfg,
        sampling_cfg=sampling_cfg,
        featurization_cfg=featurization_cfg,
        template_data=template_data,
        motion=motion,
        parent_sorting_hdf5_path=getattr(sorting, "parent_h5_path", None),
    )
    sorting = run_peeler(
        matching_peeler,
        output_directory=output_dir,
        hdf5_filename=hdf5_filename,
        model_subdir=model_subdir,
        featurization_cfg=featurization_cfg,
        chunk_starts_samples=chunk_starts_samples,
        stop_after_n_spikes=stop_after_n_spikes,
        ensure_coverage=ensure_coverage,
        overwrite=overwrite,
        residual_filename=residual_filename,
        skip_resid_snips=skip_resid_snips,
        show_progress=show_progress,
        computation_cfg=computation_cfg,
    )

    del matching_peeler
    cleanup_and_log_gpu_usage(computation_cfg, f"Post match ({hdf5_filename}):")

    return sorting


def grab(
    output_dir: str | Path,
    recording: BaseRecording,
    sorting: DARTsortSorting,
    waveform_cfg: WaveformConfig = default_waveform_cfg,
    featurization_cfg: FeaturizationConfig = default_featurization_cfg,
    sampling_cfg: FitSamplingConfig = default_peeling_fit_sampling_cfg,
    chunk_starts_samples=None,
    overwrite=False,
    show_progress=True,
    hdf5_filename="grab.h5",
    model_subdir="grab_models",
    computation_cfg: ComputationConfig | None = None,
) -> DARTsortSorting:
    output_dir = ensure_path(output_dir)
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
    waveform_cfg: WaveformConfig = default_waveform_cfg,
    thresholding_cfg: ThresholdingConfig = default_thresholding_cfg,
    featurization_cfg: FeaturizationConfig = default_featurization_cfg,
    featurization_pipeline: WaveformPipeline | None = None,
    sampling_cfg: FitSamplingConfig = default_peeling_fit_sampling_cfg,
    extract_channel_index: Tensor | None = None,
    chunk_starts_samples=None,
    stop_after_n_spikes: int | None = None,
    ensure_coverage: float | None = None,
    overwrite=False,
    show_progress=True,
    hdf5_filename="threshold.h5",
    model_subdir="threshold_models",
    computation_cfg: ComputationConfig | None = None,
) -> DARTsortSorting:
    output_dir = ensure_path(output_dir)
    computation_cfg = ensure_computation_config(computation_cfg)
    thresholder = Threshold.from_config(
        recording=recording,
        waveform_cfg=waveform_cfg,
        thresholding_cfg=thresholding_cfg,
        featurization_cfg=featurization_cfg,
        featurization_pipeline=featurization_pipeline,
        extract_channel_index=extract_channel_index,
        sampling_cfg=sampling_cfg,
    )
    sorting = run_peeler(
        thresholder,
        output_directory=output_dir,
        hdf5_filename=hdf5_filename,
        model_subdir=model_subdir,
        featurization_cfg=featurization_cfg,
        chunk_starts_samples=chunk_starts_samples,
        stop_after_n_spikes=stop_after_n_spikes,
        ensure_coverage=ensure_coverage,
        overwrite=overwrite,
        show_progress=show_progress,
        computation_cfg=computation_cfg,
    )

    del thresholder
    cleanup_and_log_gpu_usage(computation_cfg, f"Post threshold ({hdf5_filename}):")

    return sorting


def cluster(
    recording: BaseRecording,
    sorting: DARTsortSorting,
    motion: MotionInfo,
    clustering_cfg: ClusteringConfig | None = default_clustering_cfg,
    clustering_features_cfg: (
        ClusteringFeaturesConfig | None
    ) = default_clustering_features_cfg,
    refinement_cfgs: Sequence[RefinementConfig | None] | None = None,
    computation_cfg: ComputationConfig | None = None,
    features: SimpleMatrixFeatures | None = None,
    *,
    _save_cfg: DARTsortInternalConfig | None = None,
    _save_initial_name="initial",
    _save_refined_name_fmt="refined0{stepname}",
    _save_dir=None,
):
    computation_cfg = ensure_computation_config(computation_cfg)
    if features is None:
        assert clustering_features_cfg is not None
        features = SimpleMatrixFeatures.from_config(
            sorting=sorting,
            motion=motion,
            clustering_features_cfg=clustering_features_cfg,
            computation_cfg=computation_cfg,
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
    if clusterer.needs_stable_features():
        assert clustering_features_cfg is not None
        stable_features = StableWaveformFeatures.from_config(
            sorting=sorting,
            motion=motion,
            clustering_features_cfg=clustering_features_cfg,
            computation_cfg=computation_cfg,
        )
    else:
        stable_features = None

    result = clusterer.cluster(
        recording=recording,
        sorting=sorting,
        features=features,
        stable_features=stable_features,
        motion=motion,
    )

    del features, clusterer
    cleanup_and_log_gpu_usage(computation_cfg, "Post cluster:")

    return result
