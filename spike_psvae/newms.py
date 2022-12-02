import numpy as np
from spike_psvae import (
    spike_train_utils,
    before_deconv_merge_split,
    deconv_resid_merge,
)
from scipy.spatial.distance import cdist


def registered_maxchan(
    spike_index, p, geom, pfs=30000, offset=None, depth_domain=None, ymin=0
):
    pos = geom[spike_index[:, 1]]
    if p.ndim == 1 or p.shape[1] == 1:
        if offset is None:
            p = p - np.median(p)
        pos[:, 1] -= p[spike_index[:, 0] // pfs]
    elif p.ndim == 2:
        from spike_psvae.ibme import warp_nonrigid

        pos[:, 1] -= ymin
        print(p.shape, depth_domain.shape)
        pos[:, 1] = warp_nonrigid(
            pos[:, 1], spike_index[:, 0] / pfs, p, depth_domain=depth_domain
        )
    regmc = cdist(pos, geom).argmin(1)
    return regmc


def new_merge_split(
    spike_train,
    n_channels,
    raw_bin,
    sub_h5,
    geom,
    outdir,
    n_workers=1,
    herding_npcs=2,
    herding_clust="hdbscan",
    merge_resid_threshold=2.0,
):
    (
        aligned_spike_train,
        order,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        spike_train,
        n_channels,
        raw_bin,
    )

    new_labels = before_deconv_merge_split.split_clusters(
        aligned_spike_train[:, 1],
        raw_bin,
        sub_h5,
        n_workers=n_workers,
    )

    (
        aligned_spike_train2,
        order,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        np.c_[aligned_spike_train[:, 0], new_labels],
        n_channels,
        raw_bin,
    )

    assert (order == np.arange(len(order))).all()

    print("Save split...")
    np.save(outdir / "split_st.npy", aligned_spike_train2)
    np.save(outdir / "split_templates.npy", templates)
    np.save(outdir / "split_order.npy", order)

    aligned_times, new_labels = before_deconv_merge_split.merge_clusters(
        sub_h5,
        raw_bin,
        aligned_spike_train2[:, 1],
        templates,
    )

    (
        aligned_spike_train3,
        order,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        np.c_[aligned_spike_train2[:, 0], new_labels],
        n_channels,
        raw_bin,
        max_shift=10,
    )

    kept = aligned_spike_train3[:, 1] >= 0
    times_updated, labels_updated = deconv_resid_merge.run_deconv_merge(
        aligned_spike_train3[kept],
        geom,
        raw_bin,
        templates.ptp(1).argmax(1),
        merge_resid_threshold=merge_resid_threshold,
    )
    aligned_spike_train3[kept, 0] = times_updated
    aligned_spike_train3[kept, 1] = labels_updated

    (
        aligned_spike_train4,
        reorder,
        templates,
        template_shifts,
    ) = spike_train_utils.clean_align_and_get_templates(
        aligned_spike_train3,
        n_channels,
        raw_bin,
        max_shift=20,
    )
    order = order[reorder]

    print("Save merge...")
    np.save(outdir / "merge_st.npy", aligned_spike_train4)
    np.save(outdir / "merge_templates.npy", templates)
    np.save(outdir / "merge_order.npy", order)

    return aligned_spike_train4, templates, order