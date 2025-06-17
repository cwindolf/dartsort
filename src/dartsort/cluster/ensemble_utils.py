import numpy as np
from tqdm.auto import tqdm

from . import merge, split


def split_merge_ensemble(
    recording,
    chunk_sortings,
    motion_est=None,
    split_config=None,
    merge_config=None,
    merge_template_config=None,
    n_jobs_split=0,
):
    # split inside each chunk
    chunk_sortings = [
        split.split_clusters(
            sorting,
            split_strategy=split_config.split_strategy,
            recursive=split_config.recursive_split,
            n_jobs=n_jobs_split,
            motion_est=motion_est,
            show_progress=False,
        )
        for sorting in tqdm(chunk_sortings, desc="Split within chunks")
    ]

    # merge within and across chunks
    sorting = merge.merge_across_sortings(
        chunk_sortings,
        recording,
        template_config=merge_template_config,
        motion_est=motion_est,
        cross_merge_distance_threshold=merge_config.cross_merge_distance_threshold,
        within_merge_distance_threshold=merge_config.merge_distance_threshold,
        min_spatial_cosine=merge_config.min_spatial_cosine,
        show_progress=True,
    )

    return sorting
