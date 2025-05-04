from dataclasses import asdict
from logging import getLogger
import pickle
import shutil
from pathlib import Path

import numpy as np

from dartsort.util.py_util import resolve_path
from dartsort.util.data_util import DARTsortSorting
from dartsort.util.internal_config import DARTsortInternalConfig

logger = getLogger(__name__)


def ds_save_intermediate_labels(
    step_name: str,
    step_sorting: DARTsortSorting,
    output_dir: Path | str,
    cfg: DARTsortInternalConfig,
    step_labels: np.ndarray | None = None,
    work_dir: str | Path | None = None,
):
    if not cfg.save_intermediate_labels:
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
        shutil.copy2(step_labels_npy, targ_labels_npy)


def ds_dump_config(internal_cfg: DARTsortInternalConfig, output_dir: Path):
    import json

    json_path = output_dir / "_dartsort_internal_config.json"
    with open(json_path, "w") as jsonf:
        json.dump(asdict(internal_cfg), jsonf)
    logger.dartsortdebug(f"Recorded config to {json_path}.")


def ds_all_to_workdir(output_dir: Path, work_dir: Path | None = None, overwrite=False):
    if work_dir is None:
        return
    if overwrite:
        logger.dartsortdebug(f"Working in {work_dir}. No copy since {overwrite=}.")
        return
    # TODO: maybe no need to copy everything, esp. if fast forwarding?
    logger.dartsortdebug(f"Copy {output_dir=} -> {work_dir=}.")
    shutil.copytree(output_dir, work_dir, symlinks=True, dirs_exist_ok=True)


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
    if not (cfg.keep_initial_features or is_final):
        return

    # find h5 and models and copy
    assert sorting.parent_h5_path is not None
    h5_path = resolve_path(sorting.parent_h5_path)
    assert h5_path.exists()
    models_path = h5_path.parent / f"{h5_path.stem}_models"

    targ_h5 = output_dir / h5_path.name
    logger.dartsortdebug(f"Copy intermediate {h5_path=} -> {targ_h5=}.")
    shutil.copy2(h5_path, targ_h5, follow_symlinks=False)

    if models_path.exists():
        targ_models = output_dir / models_path.name
        logger.dartsortdebug(f"Copy intermediate {models_path=} -> {targ_models=}.")
        shutil.copytree(models_path, targ_models, symlinks=True, dirs_exist_ok=True)


def ds_handle_delete_intermediate_features(
    cfg: DARTsortInternalConfig,
    final_sorting: DARTsortSorting,
    output_dir: Path,
    work_dir: Path | None = None,
):
    if work_dir is not None:
        # they'll get deleted anyway and were not copied
        return
    if cfg.keep_initial_features:
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
