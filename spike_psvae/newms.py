import numpy as np
from spike_psvae import (
    spike_train_utils,
    before_deconv_merge_split,
    deconv_resid_merge,
)
from scipy.spatial.distance import cdist
from functools import partial


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
    merge_resid_threshold=2.0,
    threshold_diptest=0.5,
    relocated=False,
    trough_offset=42,
    spike_length_samples=121,
    extra_pc_split=True,
    pc_only=False,
    load_split=False,
):
    if not load_split:
        (
            aligned_spike_train,
            order,
            templates,
            template_shifts,
        ) = spike_train_utils.clean_align_and_get_templates(
            spike_train,
            n_channels,
            raw_bin,
            trough_offset=trough_offset,
            spike_length_samples=spike_length_samples,
        )

        # yizi_split = partial(before_deconv_merge_split.herding_split, clusterer="optics")
        # yizi_split.__name__ = "optics Split"

        if extra_pc_split:
            new_labels = before_deconv_merge_split.split_clusters(
                aligned_spike_train[:, 1],
                raw_bin,
                sub_h5,
                n_workers=n_workers,
                relocated=relocated,
                split_steps=(
                    before_deconv_merge_split.herding_split,
                    before_deconv_merge_split.herding_split,
                ),
                recursive_steps=(False, True),
                split_step_kwargs=(
                    {},
                    dict(
                        use_features=False,
                        n_pca_features=3,
                        hdbscan_kwargs=dict(min_cluster_size=15, min_samples=5),
                    ),
                ),
                # split_steps=(before_deconv_merge_split.herding_split, yizi_split,),
                # recursive_steps=(False, True,),
                # split_steps=(yizi_split,),
                # recursive_steps=(True,),
            )
        elif pc_only:
            new_labels = before_deconv_merge_split.split_clusters(
                aligned_spike_train[:, 1],
                raw_bin,
                sub_h5,
                n_workers=n_workers,
                relocated=relocated,
                split_steps=(before_deconv_merge_split.herding_split,),
                recursive_steps=(True,),
                split_step_kwargs=(
                    dict(
                        use_features=False,
                        n_pca_features=3,
                        hdbscan_kwargs=dict(min_cluster_size=25, min_samples=5),
                    ),
                ),
                # split_steps=(before_deconv_merge_split.herding_split, yizi_split,),
                # recursive_steps=(False, True,),
                # split_steps=(yizi_split,),
                # recursive_steps=(True,),
            )
        else:
            new_labels = before_deconv_merge_split.split_clusters(
                aligned_spike_train[:, 1],
                raw_bin,
                sub_h5,
                n_workers=n_workers,
                relocated=relocated,
                split_steps=(before_deconv_merge_split.herding_split,),
                recursive_steps=(True,),
                # split_steps=(before_deconv_merge_split.herding_split, yizi_split,),
                # recursive_steps=(False, True,),
                # split_steps=(yizi_split,),
                # recursive_steps=(True,),
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
            trough_offset=trough_offset,
            spike_length_samples=spike_length_samples,
            order_units_by_z=True,
            geom=geom,
        )

        assert (order == np.arange(len(order))).all()

        print("Save split...")
        np.save(outdir / "split_st.npy", aligned_spike_train2)
        np.save(outdir / "split_templates.npy", templates)
        np.save(outdir / "split_order.npy", order)
    else:
        aligned_spike_train2 = np.load(outdir / "split_st.npy")
        templates = np.load(outdir / "split_templates.npy")
        order = np.load(outdir / "split_order.npy")

    aligned_times, new_labels = before_deconv_merge_split.merge_clusters(
        sub_h5,
        raw_bin,
        aligned_spike_train2[:, 1],
        templates,
        relocated=relocated,
        trough_offset=trough_offset,
        n_jobs=n_workers,
        threshold_diptest=threshold_diptest,
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
        max_shift=20,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
    )

    kept = aligned_spike_train3[:, 1] >= 0
    times_updated, labels_updated = deconv_resid_merge.run_deconv_merge(
        aligned_spike_train3[kept],
        geom,
        raw_bin,
        templates.ptp(1).argmax(1),
        merge_resid_threshold=merge_resid_threshold,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
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
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
        order_units_by_z=True,
        geom=geom,
    )
    order = order[reorder]

    print("Save merge...")
    np.save(outdir / "merge_st.npy", aligned_spike_train4)
    np.save(outdir / "merge_templates.npy", templates)
    np.save(outdir / "merge_order.npy", order)

    return aligned_spike_train4, templates, order
