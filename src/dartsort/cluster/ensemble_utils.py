import numpy as np
from tqdm.auto import tqdm

from ..util.internal_config import default_split_merge_config
from . import merge, split


def get_indices_in_chunk(times_s, chunk_time_range_s):
    if chunk_time_range_s is None:
        return np.arange(len(times_s))

    return np.flatnonzero(
        (times_s >= chunk_time_range_s[0]) & (times_s < chunk_time_range_s[1])
    )


def split_merge_ensemble(
    recording,
    chunk_sortings,
    motion_est=None,
    split_merge_config=default_split_merge_config,
    n_jobs_split=0,
    n_jobs_merge=0,
    device=None,
    show_progress=True,
):
    # split inside each chunk
    chunk_sortings = [
        split.split_clusters(
            sorting,
            split_strategy=split_merge_config.split_strategy,
            recursive=split_merge_config.recursive_split,
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
        template_config=split_merge_config.merge_template_config,
        motion_est=motion_est,
        cross_merge_distance_threshold=split_merge_config.cross_merge_distance_threshold,
        within_merge_distance_threshold=split_merge_config.merge_distance_threshold,
        min_spatial_cosine=split_merge_config.min_spatial_cosine,
        device=device,
        n_jobs=n_jobs_merge,
        n_jobs_templates=n_jobs_merge,
        show_progress=True,
    )

    return sorting
