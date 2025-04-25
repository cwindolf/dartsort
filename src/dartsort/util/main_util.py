from logging import getLogger

import numpy as np

from dartsort.util.py_util import resolve_path

logger = getLogger(__name__)


def ds_tasks(step_name, step_sorting, output_dir, cfg, step_labels=None):
    output_dir = resolve_path(output_dir, strict=True)

    if cfg.save_intermediate_labels:
        step_labels_npy = output_dir / f"{step_name}_labels.npy"
        logger.info(f"Saving {step_name} labels to {step_labels_npy}")
        if step_labels is None:
            step_labels = step_sorting.labels
        np.save(step_labels_npy, step_labels, allow_pickle=False)
