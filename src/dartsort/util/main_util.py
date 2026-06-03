import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import numpy as np
from spikeinterface.core import BaseRecording

from ..util.data_util import DARTsortSorting
from ..util.internal_config import (
    ClusteringConfig,
    ClusteringFeaturesConfig,
    DARTsortInternalConfig,
    FeaturizationConfig,
    FitSamplingConfig,
    RefinementConfig,
)
from ..util.logging_util import get_logger
from ..util.motion import MotionInfo, try_load_motion_info
from ..util.py_util import dartcopy2, dartcopytree, ensure_path

logger = get_logger(__name__)


def ds_save_intermediate_sorting(
    step_name: str,
    step_sorting: DARTsortSorting,
    output_dir: Path | str | None,
    cfg: DARTsortInternalConfig | None,
    work_dir: str | Path | None = None,
):
    if cfg is not None and not cfg.save_intermediate_labels:
        return
    if output_dir is None:
        return
    output_dir = ensure_path(output_dir, strict=True)
    if work_dir is None:
        store_dir = output_dir
    else:
        store_dir = ensure_path(work_dir, strict=True)

    step_npz = store_dir / f"{step_name}.npz"
    logger.info(f"Saving {step_name} labels to {step_npz}")
    logger.info(f"{step_name}: {step_sorting}.")
    step_sorting.save(step_npz)

    if work_dir is not None:
        targ_npz = output_dir / step_npz.name
        logger.info(f"Copy {step_npz} -> {targ_npz}.")
        dartcopy2(cfg, step_npz, targ_npz)


def ds_save_intermediate_labels(
    step_name: str,
    step_sorting: DARTsortSorting,
    output_dir: Path | str | None,
    cfg: DARTsortInternalConfig | None,
    step_labels: np.ndarray | None = None,
    work_dir: str | Path | None = None,
):
    if cfg is not None and not cfg.save_intermediate_labels:
        return
    if output_dir is None:
        return
    output_dir = ensure_path(output_dir, strict=True)
    if work_dir is None:
        store_dir = output_dir
    else:
        store_dir = ensure_path(work_dir, strict=True)

    step_labels_npy = store_dir / f"{step_name}_labels.npy"
    logger.info(f"Saving {step_name} labels to {step_labels_npy}")
    logger.info(f"{step_name}: {step_sorting}.")
    if step_labels is None:
        step_labels = step_sorting.labels
    if step_labels is None:
        raise ValueError(f"No step labels to save at step {step_name}.")
    np.save(step_labels_npy, step_labels, allow_pickle=False)

    if work_dir is not None:
        targ_labels_npy = output_dir / step_labels_npy.name
        logger.info(f"Copy {step_labels_npy} -> {targ_labels_npy}.")
        dartcopy2(cfg, step_labels_npy, targ_labels_npy)


def ds_dump_config(internal_cfg: DARTsortInternalConfig, output_dir: Path):
    json_path = output_dir / "_dartsort_internal_config.json"
    with open(json_path, "w") as jsonf:
        json.dump(asdict(internal_cfg), jsonf)
    logger.info(f"Recorded config to {json_path}.")


def ds_all_to_workdir(
    *,
    internal_cfg: DARTsortInternalConfig,
    output_dir: Path,
    work_dir: Path | None = None,
    recording: BaseRecording,
    overwrite=False,
    rec_subdir="recppx",
    sort_subdir="dartsort",
) -> tuple[BaseRecording, Path | None]:
    """Copy stuff to temporary working directory, if there is one."""
    if work_dir is None:
        return recording, None

    if recording is not None and internal_cfg.copy_recording_to_tmpdir:
        rec_dir = work_dir / rec_subdir
        logger.info(
            f"Writing recording to {rec_dir}. Parallelism is handled by "
            "spikeinterface for this part, so use its `set_global_job_kwargs()` "
            "function if you're waiting around."
        )
        recording = recording.save_to_folder(str(rec_dir))

    sort_dir = work_dir / sort_subdir
    if overwrite:
        logger.info(f"Working in {work_dir}. No copy since {overwrite=}.")
        return recording, sort_dir

    # TODO: maybe no need to copy everything, esp. if fast forwarding?
    logger.dartsortdebug(f"Copy {output_dir=} -> {sort_dir=}.")
    dartcopytree(internal_cfg, output_dir, sort_dir)
    return recording, sort_dir


def ds_save_motion(
    motion: MotionInfo,
    output_dir: Path,
    work_dir: Path | None = None,
    overwrite: bool = False,
):
    if work_dir is None:
        return
    motion.save(output_directory=output_dir, overwrite=overwrite)


def motion_needs_peaks(
    cfg: DARTsortInternalConfig, recording: BaseRecording, sorting: DARTsortSorting
):
    if cfg.subsampling_presence == 1.0:
        return False
    if cfg.subsampling_spikes is None:
        return False

    # assert sorting's chunk starts, sorted, match full recording's
    # so, this means sorting could have been shuffled
    # this function not designed for that kind of sorting
    targ_chunk_starts = np.arange(
        0, recording.get_num_samples(), cfg.initial_detection_cfg.chunk_length_samples
    )
    my_chunk_starts = sorting._load_dataset("chunk_starts_samples")
    assert np.array_equal(targ_chunk_starts, np.sort(my_chunk_starts))

    # check if sorting quit early
    last_chunk_start = sorting._load_dataset("last_chunk_start").item()
    complete = my_chunk_starts[-1] == last_chunk_start

    return not complete


def ds_handle_link_from(cfg: DARTsortInternalConfig, output_dir: Path):
    if cfg.link_from is None:
        return

    link_from = ensure_path(cfg.link_from, strict=True, resolve=True)
    assert link_from.is_dir()

    link_patterns = []
    link_matching1 = cfg.link_step == "matching1"
    link_refined0 = link_matching1 or cfg.link_step == "refined0"
    link_detection = link_refined0 or cfg.link_step == "detection"
    link_denoising = link_detection or cfg.link_step == "denoising"

    if link_denoising:
        link_patterns.extend(["subtraction_models/*denoising_pipeline.pt"])
    if link_detection:
        link_patterns.extend(
            [
                "subtraction.h5",
                "motion.pkl",
                "motionthreshold.h5",
                "subtraction_models/featurization_pipeline.pt",
            ]
        )
    if link_refined0:
        link_patterns.extend(["initial*.npy", "refined0*.npy"])
    if link_matching1:
        link_patterns.extend(["matching1.h5", "matching1_models"])

    for pattern in link_patterns:
        for src in link_from.glob(pattern):
            rel_part = src.relative_to(link_from)
            targ = output_dir / rel_part
            targ.parent.mkdir(exist_ok=True)
            if targ.exists():
                logger.dartsortdebug(f"{targ} exists, won't link.")
                continue
            if not src.exists():
                logger.dartsortdebug(f"{src} doesn't exist, won't link.")
                continue
            logger.dartsortdebug(f"Link {targ} -> {src}.")
            targ.symlink_to(src)


def ds_save_features(
    cfg: DARTsortInternalConfig | None,
    sorting: DARTsortSorting,
    output_dir: Path,
    work_dir: Path | None = None,
    is_final=False,
    ensure_saving: bool | None = None,
):
    if work_dir is None:
        # nothing to copy
        return

    if ensure_saving is None:
        assert cfg is not None
        if not (cfg.save_intermediate_features or is_final):
            return
    elif not ensure_saving:
        return

    # find h5 and models and copy
    assert sorting.parent_h5_path is not None
    h5_path = ensure_path(sorting.parent_h5_path)
    assert h5_path.exists()
    models_path = h5_path.parent / f"{h5_path.stem}_models"

    targ_h5 = output_dir / h5_path.name
    logger.dartsortdebug(f"Copy intermediate {h5_path=} -> {targ_h5=}.")
    dartcopy2(cfg, h5_path, targ_h5)

    if models_path.exists():
        targ_models = output_dir / models_path.name
        pconv_h5 = targ_models / "pconv.h5"
        if cfg is not None and cfg.matching_cfg.delete_pconv and pconv_h5.exists():
            pconv_h5.unlink()
        elif cfg is None:
            assert not pconv_h5.exists()  # don't know what to do with it, pass cfg
        logger.dartsortdebug(f"Copy intermediate {models_path=} -> {targ_models=}.")
        dartcopytree(cfg, models_path, targ_models)


def ds_handle_delete_intermediate_features(
    cfg: DARTsortInternalConfig,
    final_sorting: DARTsortSorting,
    output_dir: Path,
    work_dir: Path | None = None,
):
    if work_dir is not None:
        # they'll get deleted anyway and were not copied
        return
    if cfg.save_intermediate_features:
        return

    # find all non-final h5s, models and delete them
    assert final_sorting.parent_h5_path is not None
    final_h5 = ensure_path(final_sorting.parent_h5_path)
    assert final_h5.exists()
    assert final_h5.parent == output_dir

    for h5_path in output_dir.glob("*.h5"):
        if h5_path == final_h5:
            continue
        assert h5_path.name != final_h5.name

        h5_path = output_dir / h5_path.name
        models_path = output_dir / f"{h5_path.stem}_models"

        logger.dartsortdebug(f"Clean up: remove {h5_path=}.")
        h5_path.unlink()
        if models_path.exists():
            assert models_path.is_dir()
            shutil.rmtree(models_path)


def ds_fast_forward(
    store_dir: Path, cfg: DARTsortInternalConfig
) -> tuple[int, DARTsortSorting | None, MotionInfo | None]:
    """Fast-forward to the where sorting left off

    # TODO: error if there is a saved cfg which differs? Maybe just optionally?

    Returns
    -------
    next_step: int
    cur_sorting: DARTsortSorting
    """
    # this cur_h5 variable points to the peeling result we'll try to load
    cur_h5 = sub_h5 = store_dir / "subtraction.h5"
    cur_step = 0
    if not sub_h5.exists():
        return cur_step, None, None

    # if subtraction is finished, we can try to load a motion estimate
    motion = try_load_motion_info(store_dir)
    mstr = ", with loaded motion info." if motion is not None else ""

    matching_h5s = sorted(store_dir.glob("matching*.h5"))
    for cur_step, cur_h5 in enumerate(matching_h5s, start=1):
        assert cur_h5.name == f"matching{cur_step}.h5"

    # at this point, cur_h5 is the most recent h5 that exists, and
    # current_step is the last step that was in progress.
    # now, the peeling and clustering for that step may not have
    # finished! in that case we want to resume from that step.
    # on the other hand, the step could have finished entirely --
    # but we would only know that if refined{step}_labels.npy exists,
    # and that would only happen if cfg.save_intermediate_labels.
    if cfg.save_intermediate_labels:
        cur_labels_npy = store_dir / f"refined{cur_step}_labels.npy"
        if cur_labels_npy.exists():
            labels = np.load(cur_labels_npy)
            sorting = DARTsortSorting.from_peeling_hdf5(cur_h5)
            sorting = sorting.ephemeral_replace(labels=labels)
            logger.info(
                f"Resuming at step {cur_step + 1} with previous sorting from "
                f"{cur_h5.name} and {cur_labels_npy.name}{mstr}."
            )
            return cur_step + 1, sorting, motion

    # at this point, either the last round of peeling finished, or it
    # didn't. we need to run peeler_is_done to check. but that's something
    # done internally in each step anyway. so, what we do now is get the
    # sorting for the PREVIOUS step which is the one we need to resume at
    # the CURRENT step aka cur_step.
    if cur_step == 0:
        return cur_step, None, motion

    prev_labels_npy = store_dir / f"refined{cur_step - 1}_labels.npy"
    prev_labels = None
    if prev_labels_npy.exists():
        prev_labels = np.load(prev_labels_npy)
    if cur_step == 1:
        prev_h5 = sub_h5
    else:
        prev_h5 = store_dir / f"matching{cur_step - 1}.h5"

    logger.info(
        f"Resuming at step {cur_step} with previous sorting from "
        f"{prev_h5.name} and {prev_labels_npy.name}{mstr}."
    )

    prev_sorting = DARTsortSorting.from_peeling_hdf5(prev_h5)
    if prev_labels is not None:
        prev_sorting = prev_sorting.ephemeral_replace(labels=prev_labels)

    return cur_step, prev_sorting, motion


def _matching_step_cfgs(
    is_final: bool, is_subsampling: bool, cfg: DARTsortInternalConfig
) -> tuple[
    ClusteringConfig | None,
    ClusteringFeaturesConfig,
    Sequence[RefinementConfig | None],
    FeaturizationConfig,
    FitSamplingConfig,
]:
    clus_cfg = cfg.clustering_cfg if cfg.recluster_after_first_matching else None
    gmm_as_classifier = (
        is_final and is_subsampling and cfg.refinement_cfg.refinement_strategy == "tmm"
    )
    if not cfg.final_refinement:
        gmm_clus_cfg = clus_cfg = None
        ref_cfgs = []
    elif gmm_as_classifier:
        gmm_clus_cfg = clus_cfg
        clus_cfg = None
        ref_cfgs = [cfg.agglomerate_cfg]
    else:
        gmm_clus_cfg = None
        ref_cfgs = [cfg.pre_refinement_cfg, cfg.refinement_cfg, cfg.agglomerate_cfg]
    clfeat_cfg = cfg.clustering_features_cfg

    if cfg.final_refinement and gmm_as_classifier:
        still_need_projs_saved = (
            cfg.recluster_after_first_matching or cfg.always_save_final_tpca_feature
        )
        feat_cfg = replace(
            cfg.featurization_cfg,
            save_input_tpca_projs=still_need_projs_saved,
            compute_input_tpca_projs_regardless=True,
            use_gmm_classifier=True,
            pre_gmm_clustering_cfg=gmm_clus_cfg,
            gmm_clustering_features_cfg=cfg.clustering_features_cfg,
            pre_gmm_refinement_cfgs=[cfg.pre_refinement_cfg],
            gmm_refinement_cfg=cfg.refinement_cfg,
        )
        samp_cfg = cfg.refinement_cfg.sampling_cfg
        assert clus_cfg is None
        if not still_need_projs_saved:
            clfeat_cfg = replace(cfg.clustering_features_cfg, n_main_channel_pcs=0)
    else:
        feat_cfg = cfg.featurization_cfg
        samp_cfg = cfg.peeler_sampling_cfg

    return clus_cfg, clfeat_cfg, ref_cfgs, feat_cfg, samp_cfg


def ds_save_timing(timings: dict[str, float], output_dir: Path):
    if (output_dir / "timing.json").exists():
        return
    with open(output_dir / "timing.json", "w") as jsonf:
        json.dump(timings, jsonf)
