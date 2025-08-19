from dataclasses import asdict
from logging import getLogger
import pickle
import shutil
from pathlib import Path

import numpy as np

from dartsort.util.py_util import resolve_path, dartcopy2, dartcopytree
from dartsort.util.data_util import DARTsortSorting
from dartsort.util.internal_config import DARTsortInternalConfig

logger = getLogger(__name__)


def ds_save_intermediate_labels(
    step_name: str,
    step_sorting: DARTsortSorting,
    output_dir: Path | str,
    cfg: DARTsortInternalConfig | None,
    step_labels: np.ndarray | None = None,
    work_dir: str | Path | None = None,
):
    if cfg is not None and not cfg.save_intermediate_labels:
        return
    output_dir = resolve_path(output_dir, strict=True)
    if work_dir is None:
        store_dir = output_dir
    else:
        store_dir = resolve_path(work_dir, strict=True)

    step_labels_npy = store_dir / f"{step_name}_labels.npy"
    logger.info(f"Saving {step_name} labels to {step_labels_npy}")
    if step_labels is None:
        step_labels = step_sorting.labels
    np.save(step_labels_npy, step_labels, allow_pickle=False)

    if work_dir is not None:
        targ_labels_npy = output_dir / step_labels_npy.name
        logger.dartsortdebug(f"Copy {step_labels_npy} -> {targ_labels_npy}.")
        dartcopy2(cfg, step_labels_npy, targ_labels_npy)


def ds_dump_config(internal_cfg: DARTsortInternalConfig, output_dir: Path):
    import json

    json_path = output_dir / "_dartsort_internal_config.json"
    with open(json_path, "w") as jsonf:
        json.dump(asdict(internal_cfg), jsonf)
    logger.dartsortdebug(f"Recorded config to {json_path}.")


def ds_all_to_workdir(
    internal_cfg: DARTsortInternalConfig,
    output_dir: Path,
    work_dir: Path | None = None,
    overwrite=False,
):
    if work_dir is None:
        return
    if overwrite:
        logger.dartsortdebug(f"Working in {work_dir}. No copy since {overwrite=}.")
        return
    # TODO: maybe no need to copy everything, esp. if fast forwarding?
    logger.dartsortdebug(f"Copy {output_dir=} -> {work_dir=}.")
    dartcopytree(internal_cfg, output_dir, work_dir)


def ds_save_motion_est(
    motion_est,
    output_dir: Path,
    work_dir: Path | None = None,
    overwrite=False,
):
    if work_dir is None:
        return
    if motion_est is None:
        return

    motion_est_pkl = output_dir / "motion_est.pkl"
    if overwrite or not motion_est_pkl.exists():
        with open(motion_est_pkl, "wb") as jar:
            pickle.dump(motion_est, jar)


def ds_save_features(
    cfg: DARTsortInternalConfig,
    sorting: DARTsortSorting,
    output_dir: Path,
    work_dir: Path | None = None,
    is_final=False,
):
    if work_dir is None:
        # nothing to copy
        return
    if not (cfg.save_intermediate_features or is_final):
        return

    # find h5 and models and copy
    assert sorting.parent_h5_path is not None
    h5_path = resolve_path(sorting.parent_h5_path)
    assert h5_path.exists()
    models_path = h5_path.parent / f"{h5_path.stem}_models"

    targ_h5 = output_dir / h5_path.name
    logger.dartsortdebug(f"Copy intermediate {h5_path=} -> {targ_h5=}.")
    dartcopy2(h5_path, targ_h5, follow_symlinks=False)

    if models_path.exists():
        targ_models = output_dir / models_path.name
        pconv_h5 = targ_models / "pconv.h5"
        if cfg.matching_cfg.delete_pconv and pconv_h5.exists():
            pconv_h5.unlink()
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
    final_h5 = resolve_path(final_sorting.parent_h5_path)
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


def ds_fast_forward(store_dir, cfg):
    """Fast-forward to the where sorting left off

    Returns
    -------
    next_step: int
    cur_sorting: DARTsortSorting
    """
    # if clustering labels 
    can_resume_from_clustering = cfg.save_intermediate_labels

    cur_h5 = sub_h5 = store_dir / "subtraction.h5"
    cur_step = 0
    if not sub_h5.exists():
        return cur_step, None

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
            sorting = DARTsortSorting.from_peeling_hdf5(cur_h5, labels=labels)
            logger.info(
                f"Resuming at step {cur_step + 1} with previous sorting from "
                f"{cur_h5.name} and {cur_labels_npy.name}."
            )
            return cur_step + 1, sorting

    # at this point, either the last round of peeling finished, or it
    # didn't. we need to run peeler_is_done to check. but that's something
    # done internally in each step anyway. so, what we do now is get the
    # sorting for the PREVIOUS step which is the one we need to resume at
    # the CURRENT step aka cur_step.
    if cur_step == 0:
        return cur_step, None

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
        f"{prev_h5.name} and {prev_labels_npy.name}."
    )

    prev_sorting = DARTsortSorting.from_peeling_hdf5(prev_h5, labels=prev_labels)

    return cur_step, prev_sorting
