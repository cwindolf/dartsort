import numpy as np
from tqdm.auto import trange


def forward_backward(
    recording,
    chunk_time_ranges_s,
    chunk_sortings,
    log_c=5,
    feature_scales=(1, 1, 50),
    motion_est=None,
):
    """
    Ensemble over HDBSCAN clustering
    triaging/subsampling/copying/splitting big clusters not implemented since we don't use it (so far)
    """
    if len(chunk_sortings) == 1:
        return chunk_sortings[0]

    times_seconds = chunk_sortings[0].times_seconds
    times_samples = chunk_sortings[0].times_samples
    min_time_s = chunk_time_ranges_s[0][0]
    idx_all_chunks = [get_indices_in_chunk(times_seconds, chunk_range) for chunk_range in chunk_time_ranges_s]

    # put all labels into one array
    # TODO: this does not allow for overlapping chunks.
    labels_all = np.full_like(times_samples, -1)
    for ix, sorting in zip(idx_all_chunks, chunk_sortings):
        assert labels_all[ix].max() < 0  # assert non-overlapping
        labels_all[ix] = sorting.labels[ix]

    # load features that we will need
    amps = chunk_sortings[0].denoised_amplitudes
    xyza = chunk_sortings[0].point_source_localizations
    x = xyza[:, 0]
    z_reg = z_abs = xyza[:, 2]
    if motion_est is not None:
        z_reg = motion_est.correct_s(times_seconds, z_abs)

    for k in trange(len(chunk_sortings) - 1, desc="Ensembling chunks"):
        # CHANGE THE 1 ---
        # idx_1 = np.flatnonzero(np.logical_and(times_seconds>=min_time_s, times_seconds<min_time_s+k*shift+chunk_size_s))
        idx_1 = np.flatnonzero(
            np.logical_and(
                times_seconds >= min_time_s,
                times_seconds < chunk_time_ranges_s[k][1],
            )
        )
        idx_2 = idx_all_chunks[k + 1]
        x_1 = feature_scales[0] * x[idx_1]
        x_2 = feature_scales[0] * x[idx_2]
        z_1 = feature_scales[1] * z_reg[idx_1]
        z_2 = feature_scales[1] * z_reg[idx_2]
        amps_1 = feature_scales[2] * np.log(log_c + amps[idx_1])
        amps_2 = feature_scales[2] * np.log(log_c + amps[idx_2])
        labels_1 = labels_all[idx_1].copy().astype("int")
        labels_2 = chunk_sortings[k + 1].labels[idx_2]
        unit_label_shift = int(labels_1.max() + 1)
        labels_2[labels_2 > -1] += unit_label_shift

        units_1 = np.unique(labels_1)
        units_1 = units_1[units_1 > -1]
        units_2 = np.unique(labels_2)
        units_2 = units_2[units_2 > -1]

        # FORWARD PASS

        dist_matrix = np.zeros((units_1.shape[0], units_2.shape[0]))

        # Speed up this code - this matrix can be sparse (only compute distance for "neighboring" units) - OK for now, still pretty fast
        for i in range(units_1.shape[0]):
            unit_1 = units_1[i]
            for j in range(units_2.shape[0]):
                unit_2 = units_2[j]
                feat_1 = np.c_[
                    np.median(x_1[labels_1 == unit_1]),
                    np.median(z_1[labels_1 == unit_1]),
                    np.median(amps_1[labels_1 == unit_1]),
                ]
                feat_2 = np.c_[
                    np.median(x_2[labels_2 == unit_2]),
                    np.median(z_2[labels_2 == unit_2]),
                    np.median(amps_2[labels_2 == unit_2]),
                ]
                dist_matrix[i, j] = ((feat_1 - feat_2) ** 2).sum()

        # find for chunk 2 units the closest units in chunk 1 and split chunk 1 units
        dist_forward = dist_matrix.argmin(0)
        units_, counts_ = np.unique(dist_forward, return_counts=True)

        for unit_to_split in units_[counts_ > 1]:
            units_to_match_to = (
                np.flatnonzero(dist_forward == unit_to_split) + unit_label_shift
            )
            features_to_match_to = np.c_[
                np.median(x_2[labels_2 == units_to_match_to[0]]),
                np.median(z_2[labels_2 == units_to_match_to[0]]),
                np.median(amps_2[labels_2 == units_to_match_to[0]]),
            ]
            for u in units_to_match_to[1:]:
                features_to_match_to = np.concatenate(
                    (
                        features_to_match_to,
                        np.c_[
                            np.median(x_2[labels_2 == u]),
                            np.median(z_2[labels_2 == u]),
                            np.median(amps_2[labels_2 == u]),
                        ],
                    )
                )
            spikes_to_update = np.flatnonzero(labels_1 == unit_to_split)
            x_s_to_update = x_1[spikes_to_update]
            z_s_to_update = z_1[spikes_to_update]
            amps_s_to_update = amps_1[spikes_to_update]
            for j, s in enumerate(spikes_to_update):
                # Don't update if new distance is too high?
                feat_s = np.c_[
                    x_s_to_update[j], z_s_to_update[j], amps_s_to_update[j]
                ]
                labels_1[s] = units_to_match_to[
                    ((feat_s - features_to_match_to) ** 2).sum(1).argmin()
                ]

        # Relabel labels_1 and labels_2
        for unit_to_relabel in units_:
            if counts_[np.flatnonzero(units_ == unit_to_relabel)][0] == 1:
                idx_to_relabel = np.flatnonzero(labels_1 == unit_to_relabel)
                labels_1[idx_to_relabel] = units_2[dist_forward == unit_to_relabel]

        # BACKWARD PASS

        units_not_matched = np.unique(labels_1)
        units_not_matched = units_not_matched[units_not_matched > -1]
        units_not_matched = units_not_matched[units_not_matched < unit_label_shift]

        if len(units_not_matched):
            all_units_to_match_to = (
                dist_matrix[units_not_matched].argmin(1) + unit_label_shift
            )
            for unit_to_split in np.unique(all_units_to_match_to):
                units_to_match_to = np.concatenate(
                    (
                        units_not_matched[all_units_to_match_to == unit_to_split],
                        [unit_to_split],
                    )
                )

                features_to_match_to = np.c_[
                    np.median(x_1[labels_1 == units_to_match_to[0]]),
                    np.median(z_1[labels_1 == units_to_match_to[0]]),
                    np.median(amps_1[labels_1 == units_to_match_to[0]]),
                ]
                for u in units_to_match_to[1:]:
                    features_to_match_to = np.concatenate(
                        (
                            features_to_match_to,
                            np.c_[
                                np.median(x_1[labels_1 == u]),
                                np.median(z_1[labels_1 == u]),
                                np.median(amps_1[labels_1 == u]),
                            ],
                        )
                    )
                spikes_to_update = np.flatnonzero(labels_2 == unit_to_split)
                x_s_to_update = x_2[spikes_to_update]
                z_s_to_update = z_2[spikes_to_update]
                amps_s_to_update = amps_2[spikes_to_update]
                for j, s in enumerate(spikes_to_update):
                    feat_s = np.c_[
                        x_s_to_update[j], z_s_to_update[j], amps_s_to_update[j]
                    ]
                    labels_2[s] = units_to_match_to[
                        ((feat_s - features_to_match_to) ** 2).sum(1).argmin()
                    ]

        #           Do we need to "regularize" and make sure the distance intra units after merging is smaller than the distance inter units before merging
        all_labels_1 = np.unique(labels_1)
        all_labels_1 = all_labels_1[all_labels_1 > -1]

        features_all_1 = np.c_[
            np.median(x_1[labels_1 == all_labels_1[0]]),  # WHY [1]?
            np.median(z_1[labels_1 == all_labels_1[0]]),
            np.median(amps_1[labels_1 == all_labels_1[0]]),
        ]
        for u in all_labels_1[1:]:
            features_all_1 = np.concatenate(
                (
                    features_all_1,
                    np.c_[
                        np.median(x_1[labels_1 == u]),
                        np.median(z_1[labels_1 == u]),
                        np.median(amps_1[labels_1 == u]),
                    ],
                )
            )

        distance_inter = (
            (features_all_1[:, :, None] - features_all_1.T[None]) ** 2
        ).sum(1)

        labels_12 = np.concatenate((labels_1, labels_2))
        _, labels_12[labels_12 > -1] = np.unique(
            labels_12[labels_12 > -1], return_inverse=True
        )  # Make contiguous
        idx_all = np.flatnonzero(
            times_seconds < min_time_s + chunk_time_ranges_s[k + 1][1]
        )
        labels_all = -1 * np.ones(
            times_seconds.shape[0]
        )  # discard all spikes at the end for now
        labels_all[idx_all] = labels_12.astype("int")

    return labels_all


def get_indices_in_chunk(times_s, chunk_time_range_s):
    if chunk_time_range_s is None:
        return slice(None)

    return np.flatnonzero(
        (times_s >= chunk_time_range_s[0]) & (times_s < chunk_time_range_s[1])
    )
